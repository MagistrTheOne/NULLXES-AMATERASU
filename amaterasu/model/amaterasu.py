from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from amaterasu.config.model_config import Amaterasu32BConfig
from amaterasu.config.validate_freeze import assert_frozen
from amaterasu.model.agency.gcis import GCIS
from amaterasu.model.agency.policy import select_intent
from amaterasu.model.attention.mask import block_structured_temporal_mask
from amaterasu.model.audio.encoder import AudioEncoder
from amaterasu.model.dynamics.predictor import LatentDynamics
from amaterasu.model.embodiment.ecd import ECDEncoder
from amaterasu.model.flow.matching import FlowNCES
from amaterasu.model.hpt.stack import HPTStack
from amaterasu.model.language.embeddings import LanguageEmbeddings
from amaterasu.model.language.special_tokens import NULL_INSTRUCTION_ID
from amaterasu.model.memory.system import MemorySystem
from amaterasu.model.nces.encode import NCESEncoder
from amaterasu.model.nces.tactile import TactileEncoder
from amaterasu.model.state.z import pack_z, pool_tokens
from amaterasu.model.temporal.ssm import PhysicalSSM
from amaterasu.model.vision.cache import VisualCache, empty_visual_cache
from amaterasu.model.vision.encoder import VisionEncoder
from amaterasu.tensors.modality import ModalityId
from amaterasu.tensors.z_schema import N_DYN, N_HUM, N_OBJ, N_SCENE


@dataclass
class PackedHPT:
    hidden: torch.Tensor
    modality_ids: torch.Tensor
    position_ids: torch.Tensor
    token_time: torch.Tensor
    valid: torch.Tensor


def _piece(
    hidden: torch.Tensor,
    modality: int,
    token_time: torch.Tensor,
    valid: torch.Tensor,
) -> PackedHPT:
    b, s, _ = hidden.shape
    device = hidden.device
    return PackedHPT(
        hidden=hidden,
        modality_ids=torch.full((b, s), modality, dtype=torch.int8, device=device),
        position_ids=torch.arange(s, device=device).view(1, s).expand(b, s),
        token_time=token_time,
        valid=valid,
    )


def pack_hpt(*pieces: PackedHPT) -> PackedHPT:
    hidden = torch.cat([p.hidden for p in pieces], dim=1)
    modality_ids = torch.cat([p.modality_ids for p in pieces], dim=1)
    token_time = torch.cat([p.token_time for p in pieces], dim=1)
    valid = torch.cat([p.valid for p in pieces], dim=1)
    b, s, _ = hidden.shape
    position_ids = torch.arange(s, device=hidden.device).view(1, s).expand(b, s)
    return PackedHPT(hidden, modality_ids, position_ids, token_time, valid)


class Amaterasu32B(nn.Module):
    """Universal AMATERASU-32B v0.1. Adapter modules are not attached."""

    def __init__(self, cfg: Amaterasu32BConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or Amaterasu32BConfig()
        assert_frozen(cfg)
        self.cfg = cfg
        self.embeddings = LanguageEmbeddings(cfg)
        self.vision = VisionEncoder(cfg)
        self.audio = AudioEncoder(cfg)
        self.nces = NCESEncoder(cfg)
        self.tactile = TactileEncoder(cfg)
        self.ecd = ECDEncoder(cfg)
        self.hpt = HPTStack(cfg)
        self.ssm = PhysicalSSM(cfg)
        self.memory = MemorySystem(cfg)
        self.dynamics = LatentDynamics(cfg)
        self.eac = GCIS(cfg)
        self.flow = FlowNCES(cfg)

    def encode_sensors(
        self,
        nces_feat: torch.Tensor,
        nces_valid: torch.Tensor,
        ecd_raw: torch.Tensor,
        ecd_topo: torch.Tensor,
        tactile: torch.Tensor | None = None,
        tactile_valid: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        camera_valid_mask: torch.Tensor | None = None,
        frame_times: torch.Tensor | None = None,
        audio_mel: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        visual_cache: VisualCache | None = None,
        refresh_vision: bool = False,
        refresh_audio: bool = False,
    ) -> dict[str, torch.Tensor | VisualCache | None]:
        hpt_nces = self.nces(nces_feat, nces_valid)
        hpt_ecd = self.ecd(ecd_raw, ecd_topo)
        b = nces_feat.shape[0]
        device = nces_feat.device
        if tactile is None:
            tactile = nces_feat.new_zeros(b, self.cfg.tactile_in)
            tactile_valid = torch.zeros(b, dtype=torch.bool, device=device)
        assert tactile_valid is not None
        hpt_tactile = self.tactile(tactile, tactile_valid)
        hpt_vision = None
        vision_mask = None
        vision_time = None
        if refresh_vision:
            if video is None or camera_valid_mask is None or frame_times is None:
                raise ValueError("FAST_SENSOR_REFRESH requires video, camera_valid_mask, frame_times")
            hpt_vision, vision_mask, vision_time = self.vision(video, camera_valid_mask, frame_times)
            if visual_cache is None:
                visual_cache = empty_visual_cache(b, hpt_vision.shape[1], self.cfg.d_model, device)
            visual_cache = visual_cache.replace(hpt_vision, vision_mask, vision_time)
        elif visual_cache is not None:
            hpt_vision, vision_mask, vision_time = (
                visual_cache.hpt_vision,
                visual_cache.vision_mask,
                visual_cache.vision_time,
            )
        hpt_audio = None
        audio_out_mask = None
        if refresh_audio:
            if audio_mel is None or audio_mask is None:
                raise ValueError("audio refresh requires audio_mel and audio_mask")
            hpt_audio, audio_out_mask = self.audio(audio_mel, audio_mask)
        return {
            "hpt_nces": hpt_nces,
            "hpt_ecd": hpt_ecd,
            "hpt_tactile": hpt_tactile,
            "hpt_vision": hpt_vision,
            "vision_mask": vision_mask,
            "vision_time": vision_time,
            "hpt_audio": hpt_audio,
            "audio_mask": audio_out_mask,
            "visual_cache": visual_cache,
        }

    def pack_fast(
        self,
        encoded: dict[str, torch.Tensor | VisualCache | None],
        nces_valid: torch.Tensor,
        nces_time: torch.Tensor,
    ) -> PackedHPT:
        hpt_ecd = encoded["hpt_ecd"]
        hpt_nces = encoded["hpt_nces"]
        hpt_tactile = encoded["hpt_tactile"]
        assert isinstance(hpt_ecd, torch.Tensor)
        assert isinstance(hpt_nces, torch.Tensor)
        assert isinstance(hpt_tactile, torch.Tensor)
        b = hpt_nces.shape[0]
        device = hpt_nces.device
        ecd_time = nces_time.new_zeros(b, hpt_ecd.shape[1])
        tac_time = nces_time.new_zeros(b, 1)
        ecd_valid = torch.ones(b, hpt_ecd.shape[1], dtype=torch.bool, device=device)
        tac_valid = torch.ones(b, 1, dtype=torch.bool, device=device)
        pieces = [
            _piece(hpt_ecd, int(ModalityId.PHYSICAL), ecd_time, ecd_valid),
            _piece(hpt_nces, int(ModalityId.PHYSICAL), nces_time, nces_valid),
            _piece(hpt_tactile, int(ModalityId.PHYSICAL), tac_time, tac_valid),
        ]
        if encoded["hpt_vision"] is not None:
            vis = encoded["hpt_vision"]
            vmask = encoded["vision_mask"]
            vtime = encoded["vision_time"]
            assert isinstance(vis, torch.Tensor) and isinstance(vmask, torch.Tensor) and isinstance(vtime, torch.Tensor)
            pieces.append(_piece(vis, int(ModalityId.VISION), vtime, vmask))
        if encoded["hpt_audio"] is not None:
            aud = encoded["hpt_audio"]
            amask = encoded["audio_mask"]
            assert isinstance(aud, torch.Tensor) and isinstance(amask, torch.Tensor)
            atime = nces_time.new_zeros(b, aud.shape[1])
            pieces.append(_piece(aud, int(ModalityId.PHYSICAL), atime, amask))
        return pack_hpt(*pieces)

    def pack_slow(self, fast_pack: PackedHPT, lang: torch.Tensor, lang_mask: torch.Tensor, lang_time: torch.Tensor) -> PackedHPT:
        lang_piece = _piece(lang, int(ModalityId.LANGUAGE), lang_time, lang_mask)
        agency_seed = self.eac.agency_tokens.unsqueeze(0).expand(lang.shape[0], -1, -1)
        agency_time = lang_time.new_zeros(lang.shape[0], agency_seed.shape[1])
        agency_valid = torch.ones(lang.shape[0], agency_seed.shape[1], dtype=torch.bool, device=lang.device)
        agency_piece = _piece(agency_seed, int(ModalityId.AGENCY), agency_time, agency_valid)
        return pack_hpt(fast_pack, lang_piece, agency_piece)

    def forward_mode(
        self,
        mode: str,
        nces_feat: torch.Tensor,
        nces_valid: torch.Tensor,
        ecd_raw: torch.Tensor,
        ecd_topo: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        lang_mask: torch.Tensor | None = None,
        tactile: torch.Tensor | None = None,
        tactile_valid: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        camera_valid_mask: torch.Tensor | None = None,
        frame_times: torch.Tensor | None = None,
        audio_mel: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        visual_cache: VisualCache | None = None,
        ssm_state: list[torch.Tensor] | None = None,
        nces_time: torch.Tensor | None = None,
        nces_traj: torch.Tensor | None = None,
        horizon_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        flow_t: torch.Tensor | None = None,
        emit_language: bool = False,
        episode_reset: torch.Tensor | None = None,
        tick: int = 0,
        memory_state=None,
        contact_idx: torch.Tensor | None = None,
        contact_valid: torch.Tensor | None = None,
    ) -> dict[str, object]:
        refresh_vision = mode in {"FAST_SENSOR_REFRESH", "TRAIN"} and video is not None
        refresh_audio = mode in {"FAST_SENSOR_REFRESH", "TRAIN"} and audio_mel is not None
        if mode == "FAST_SENSOR_REFRESH":
            refresh_vision = True
        encoded = self.encode_sensors(
            nces_feat,
            nces_valid,
            ecd_raw,
            ecd_topo,
            tactile,
            tactile_valid,
            video,
            camera_valid_mask,
            frame_times,
            audio_mel,
            audio_mask,
            visual_cache,
            refresh_vision=refresh_vision,
            refresh_audio=refresh_audio,
        )
        b = nces_feat.shape[0]
        device = nces_feat.device
        if nces_time is None:
            nces_time = torch.zeros(b, nces_feat.shape[1], device=device)
        packed = self.pack_fast(encoded, nces_valid, nces_time)
        run_slow = mode in {"TRAIN", "SLOW_AGENCY"}
        if run_slow:
            if input_ids is None:
                input_ids = torch.full((b, 1), NULL_INSTRUCTION_ID, dtype=torch.long, device=device)
                lang_mask = torch.ones(b, 1, dtype=torch.bool, device=device)
            assert lang_mask is not None
            lang_h = self.embeddings.encode(
                input_ids,
                torch.full(input_ids.shape, int(ModalityId.LANGUAGE), dtype=torch.long, device=device),
            )
            lang_time = nces_time.new_zeros(b, input_ids.shape[1])
            packed = self.pack_slow(packed, lang_h, lang_mask, lang_time)
        if packed.hidden.shape[1] > self.cfg.s_max:
            packed = PackedHPT(
                packed.hidden[:, : self.cfg.s_max],
                packed.modality_ids[:, : self.cfg.s_max],
                packed.position_ids[:, : self.cfg.s_max],
                packed.token_time[:, : self.cfg.s_max],
                packed.valid[:, : self.cfg.s_max],
            )
        temporal_bias = block_structured_temporal_mask(packed.token_time, packed.valid)
        hidden = packed.hidden
        if mode != "FAST_SENSOR_REFRESH":
            hidden = self.hpt(hidden, packed.modality_ids, packed.position_ids, temporal_bias, run_slow=run_slow)
            phys = packed.modality_ids == int(ModalityId.PHYSICAL)
            ssm_in = hidden * phys.unsqueeze(-1).to(hidden.dtype)
            hidden_ssm, ssm_state = self.ssm(ssm_in, ssm_state, episode_reset=episode_reset, tick=tick)
            hidden = torch.where(phys.unsqueeze(-1), hidden_ssm, hidden)
            wm = self.memory.working(hidden, packed.position_ids)
        else:
            wm = None
        out: dict[str, object] = {
            "hidden": hidden,
            "modality_ids": packed.modality_ids,
            "visual_cache": encoded["visual_cache"],
            "ssm_state": ssm_state,
            "working_memory": wm,
            "hpt_nces": encoded["hpt_nces"],
            "hpt_ecd": encoded["hpt_ecd"],
        }
        if run_slow:
            z_pool = pool_tokens(hidden, packed.valid)
            g_pool = pool_tokens(hidden, packed.modality_ids == int(ModalityId.LANGUAGE))
            eac_out = self.eac(hidden, z_pool, g_pool, packed.position_ids)
            decision = select_intent(eac_out["intent_scores"], eac_out["gate_logits"])
            eac_out.update(decision)
            aux9 = eac_out["aux_heads"]
            z = pack_z(
                hidden,
                packed.modality_ids,
                encoded["hpt_nces"] if isinstance(encoded["hpt_nces"], torch.Tensor) else hidden,
                nces_valid,
                aux9,
                contact_idx=contact_idx,
                contact_valid=contact_valid,
                n_obj=N_OBJ,
                n_hum=N_HUM,
                n_scene=N_SCENE,
                n_dyn=N_DYN,
            )
            dyn = self.dynamics(hidden, wm if wm is not None else hidden, eac_out["agency_tokens"])
            out["z"] = z
            out["eac"] = eac_out
            out["dynamics"] = dyn
            epi_read, epi_write = self.memory.episodic.compress(hidden)
            out["episodic_compressed"] = epi_write
            out["episodic_tokens"] = epi_read
            sem_q = z_pool
            out["semantic_query"] = self.memory.semantic.Wq(sem_q)
        use_flow = mode in {"FAST_HOLD", "FAST_ACT", "TRAIN"} and nces_traj is not None
        if use_flow:
            assert nces_traj is not None and horizon_mask is not None and node_mask is not None and flow_t is not None
            pred512, traj = self.flow(nces_traj, node_mask, horizon_mask, flow_t)
            out["flow_pred"] = pred512
            out["nces_traj"] = traj
        if emit_language:
            lang_pos = packed.modality_ids == int(ModalityId.LANGUAGE)
            out["lm_logits"] = self.embeddings.logits(hidden)
            out["lang_token_mask"] = lang_pos
        return out
