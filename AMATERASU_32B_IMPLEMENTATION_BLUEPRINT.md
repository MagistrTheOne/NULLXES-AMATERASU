# AMATERASU-32B v0.1 — IMPLEMENTATION BLUEPRINT

**Phase I only. No production modules in this step.**

**Author:** MagistrTheOne | NULLXES  
**Date:** 28 August 2026  
**Parent freeze:** [AMATERASU_EAFM_ARCHITECTURE_v0.1.md](AMATERASU_EAFM_ARCHITECTURE_v0.1.md)

Canonical target:

**NULLXES AMATERASU-32B v0.1**  
**Embodied Agency Foundation Model**  
**31,740,290,560 universal trainable parameters**  
**FROM SCRATCH**  
**Architecture frozen 28 August 2026**

AMATERASU-72B remains an **approved scale specification** (71,689,455,872), not an implementation target in Phases II–VII.

This blueprint is the file-by-file contract for Phase II onward. After architectural consistency verification, implementation proceeds in the listed order. If any later phase finds a contradiction with the freeze, stop and emit `ARCHITECTURE IMPLEMENTATION CONFLICT`. Do not silently change `31,740,290,560`.

---

## 0. Status of this document

| Kind | Meaning |
|---|---|
| **FROZEN** | From the 28 Aug 2026 architecture freeze. Implementation must match exactly. |
| **LEDGER-LOCKED** | Implied by the frozen parameter arithmetic; changing it changes TOTAL. |
| **IMPLEMENTATION CONTRACT** | Required to write tensors/code; not a redesign. Phase II must not start until these are approved or explicitly deferred with a freeze-compatible default. |
| **OPEN DECISION** | Freeze does not determine it. Do not invent a silent default in code. Report here. |

---

## 1. Repository tree

Root package name: `amaterasu`. Configs live outside the import package so freeze hashes are independent of training flags.

```text
NULLXES AMATERASU/
├── AMATERASU_EAFM_ARCHITECTURE_v0.1.md          # frozen research spec
├── AMATERASU_32B_IMPLEMENTATION_BLUEPRINT.md    # this file
├── CLAUDE.MD
├── pyproject.toml                               # package metadata only; no extra unused deps
├── configs/
│   ├── model/
│   │   ├── amaterasu_32b_v0.1.json              # immutable freeze; hashed
│   │   └── amaterasu_32b_v0.1.sha256
│   ├── train/
│   │   ├── stage_01_representation.json
│   │   ├── stage_02_motion.json
│   │   ├── stage_03_manipulation.json
│   │   ├── stage_04_cross_embodiment.json
│   │   ├── stage_05_robot.json
│   │   ├── stage_06_dynamics.json
│   │   ├── stage_07_agency.json
│   │   ├── stage_08_autonomous.json
│   │   └── stage_09_sim2real.json
│   ├── data/
│   │   ├── mixture_research.json
│   │   └── mixture_commercial.json
│   └── distributed/
│       ├── h100_80gb.json
│       ├── h200_141gb.json
│       └── b200.json
├── amaterasu/
│   ├── __init__.py
│   ├── constants.py                             # FROZEN_TOTAL = 31740290560
│   ├── config/
│   │   ├── __init__.py
│   │   ├── model_config.py                      # Amaterasu32BConfig frozen dataclass
│   │   ├── train_config.py
│   │   ├── data_config.py
│   │   └── validate_freeze.py                   # hash + reject silent expert/dim edits
│   ├── tensors/
│   │   ├── __init__.py
│   │   ├── dtypes.py
│   │   ├── modality.py                          # ModalityId enum
│   │   ├── nces_schema.py                       # node layout, D_NCES_IN=128
│   │   ├── ecd_schema.py
│   │   ├── z_schema.py
│   │   ├── sample.py                            # AMATERASUSample
│   │   └── batch.py                             # AMATERASUBatch
│   ├── model/
│   │   ├── __init__.py
│   │   ├── amaterasu.py                         # Amaterasu32B assembly + runtime modes
│   │   ├── init.py                              # from-scratch init; write safetensors
│   │   ├── accounting.py                        # executable ledger; FAIL if != freeze
│   │   ├── norms/
│   │   │   ├── rmsnorm.py
│   │   │   └── qk_norm.py
│   │   ├── attention/
│   │   │   ├── gqa.py
│   │   │   ├── rope.py
│   │   │   └── mask.py
│   │   ├── ffn/
│   │   │   └── swiglu.py
│   │   ├── hpt/
│   │   │   ├── fast_layer.py                    # L0–L11 Vision+Physical
│   │   │   ├── slow_layer.py                    # L12–L39 four FFNs
│   │   │   ├── dispatch.py                      # modality gather/scatter
│   │   │   └── stack.py                         # 40-layer HPT
│   │   ├── experts/
│   │   │   ├── physical_moe.py                  # L20–L39 sparse
│   │   │   ├── router.py                        # sigmoid, top-2, fp32 logits
│   │   │   └── metrics.py                       # utilization, entropy
│   │   ├── vision/
│   │   │   ├── tubelet.py
│   │   │   ├── encoder.py                       # 36L d=2048
│   │   │   └── cache.py                         # VisualCache timestamps
│   │   ├── audio/
│   │   │   └── encoder.py
│   │   ├── nces/
│   │   │   ├── encode.py
│   │   │   ├── nodes.py
│   │   │   └── convert.py                       # robot obs → NCES (hooks)
│   │   ├── embodiment/
│   │   │   ├── ecd.py
│   │   │   └── adapter.py                       # OUTSIDE 31.740B
│   │   ├── temporal/
│   │   │   └── ssm.py                           # 4-layer Mamba-style
│   │   ├── memory/
│   │   │   ├── working.py                       # 256 slots
│   │   │   ├── episodic.py
│   │   │   └── semantic.py
│   │   ├── state/
│   │   │   └── z.py                             # hybrid Z pack/unpack
│   │   ├── dynamics/
│   │   │   └── predictor.py                     # 8L d=3072
│   │   ├── agency/
│   │   │   ├── gcis.py                          # 4L decoder
│   │   │   ├── q_theta.py
│   │   │   ├── aux_heads.py                     # 9 heads
│   │   │   ├── gate.py                          # ALLOW/DEFER/BLOCK
│   │   │   └── intents.py                       # ACT/OBSERVE/HOLD/WAIT
│   │   ├── flow/
│   │   │   └── matching.py                      # 12L d=2048
│   │   └── language/
│   │       ├── embeddings.py                    # untied wte + lm_head
│   │       └── special_tokens.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collate.py
│   │   ├── license.py                           # research vs commercial filter
│   │   └── adapters/
│   │       ├── bones_seed.py
│   │       ├── hifi_umi.py
│   │       ├── robot_traj.py
│   │       ├── egocentric.py
│   │       ├── simulation.py
│   │       └── agency.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── curriculum.py
│   │   ├── loop.py
│   │   └── precision.py
│   ├── distributed/
│   │   ├── mesh.py                              # TP PP EP DP
│   │   ├── parallel.py
│   │   └── topologies.py
│   ├── checkpoint/
│   │   ├── safetensors_io.py
│   │   ├── manifest.py
│   │   └── resume.py
│   ├── inference/
│   │   ├── clocks.py
│   │   ├── fast_loop.py
│   │   └── slow_loop.py
│   └── utils/
│       ├── logging.py
│       └── profile.py
├── scripts/
│   ├── init_and_account.py                      # init → safetensors → print ledger
│   ├── train.py
│   ├── resume.py
│   └── infer_runtime.py
└── third_party/                                 # optional kernels only; no pretrained models
```

### File responsibilities (core)

Each row: **params in 31.740B?** Yes = universal freeze. No = adapter or non-weight.

| File | Responsibility | Public API | In 31.740B |
|---|---|---|---|
| `config/model_config.py` | Immutable freeze dataclass | `Amaterasu32BConfig`, `freeze_hash()` | n/a |
| `config/validate_freeze.py` | Reject dim/expert CLI mutations of 32B identity | `assert_frozen(cfg)` | n/a |
| `tensors/sample.py` | Canonical sample | `AMATERASUSample` | n/a |
| `model/norms/rmsnorm.py` | RMSNorm, no bias, weight `d` | `RMSNorm` | yes (counted in layers) |
| `model/attention/gqa.py` | GQA QKVO | `GQAAttention` | yes |
| `model/ffn/swiglu.py` | gate/up/down | `SwiGLU` | yes |
| `model/hpt/fast_layer.py` | L0–L11 | `HPTFastLayer` | yes |
| `model/hpt/slow_layer.py` | L12–L39 | `HPTSlowLayer` | yes |
| `model/experts/physical_moe.py` | sparse 8+1 top-2 | `PhysicalMoE` | yes |
| `model/vision/encoder.py` | 36L ViT-tubelet | `VisionEncoder` | yes |
| `model/vision/cache.py` | async visual latents | `VisualCache` | no (state, not extra params) |
| `model/nces/encode.py` | 6L 2048 | `NCESEncoder` | yes |
| `model/embodiment/ecd.py` | ECD encoder | `ECDEncoder` | yes |
| `model/embodiment/adapter.py` | NCES→robot | `EmbodimentAdapter` | **no** |
| `model/temporal/ssm.py` | 4L Mamba-style | `PhysicalSSM` | yes |
| `model/memory/*.py` | WM/epi/sem | `WorkingMemory`, … | yes |
| `model/dynamics/predictor.py` | 8L 3072 | `LatentDynamics` | yes |
| `model/agency/*.py` | GCIS, Qθ, gate | `EAC`, `ConstraintGate` | yes |
| `model/flow/matching.py` | flow matching | `FlowNCES` | yes |
| `model/language/embeddings.py` | wte+lm_head | `LanguageEmbeddings` | yes (lm_head in TOTAL, not default slow-active) |
| `model/accounting.py` | executable ledger | `account(model) -> Ledger` | n/a |
| `model/amaterasu.py` | assembly + modes | `Amaterasu32B.forward_mode(...)` | yes (sum) |
| `checkpoint/safetensors_io.py` | sharded safetensors | `save_model`, `load_model` | n/a |

Dependencies (acyclic): `tensors` → `norms/attention/ffn` → `hpt/experts` → `vision/nces/ecd/ssm/memory` → `state/dynamics/agency/flow` → `amaterasu` → `training/inference`. `accounting` depends on assembled `Amaterasu32B` only.

---

## 2. Module dependency graph

```text
Amaterasu32BConfig
    ↓
RMSNorm, QKNorm, RoPE, GQA, SwiGLU
    ↓
HPTFastLayer → HPTSlowLayer → PhysicalMoE
    ↓
VisionEncoder  NCESEncoder  AudioEncoder  ECDEncoder  LanguageEmbeddings
    ↓
PhysicalSSM → WorkingMemory
    ↓
EpisodicMemory  SemanticMemory  ZPack
    ↓
LatentDynamics → EAC/GCIS → Qθ → ConstraintGate
    ↓
FlowNCES ──(boundary)── EmbodimentAdapter ── Controller (not in repo weights)
```

Runtime graphs (Section 8) select subgraphs. Training may run Slow+Fast+refresh in one step with masks; it must not evaluate Language/Agency FFN on Vision tokens.

---

## 3. Canonical tensor contracts

Dtypes unless noted: activations **bf16**, router/softmax/Qθ/gate/loss **fp32**. Batch `B`. Time `t` is wall-clock seconds, not token index.

### 3.1 Modality IDs

```text
ModalityId: int8
  VISION=0 LANGUAGE=1 PHYSICAL=2 AGENCY=3
```

### 3.2 Vision

**FROZEN:** tubelet `(Tt, Th, Tw, C) = (2, 14, 14, 3)`; encoder `d_ve=2048`; project `2048→4096`.

**OPEN DECISION (must approve before Phase III):** native `T_video, H, W`, camera count `N_cam`. Freeze does not set resolution. Tubelet requires `T_video % 2 == 0`, `H % 14 == 0`, `W % 14 == 0`.

Proposed **IMPLEMENTATION CONTRACT** (not freeze; awaiting approval):

```text
video:           float32/uint8 [B, N_cam, T_video, C=3, H, W]
tubelets:        bf16 [B, N_cam, T_video/2, H/14, W/14, 2*14*14*3]
ve_tokens:       bf16 [B, N_cam * (T_video/2) * (H/14) * (W/14), 2048]
hpt_vision:      bf16 [B, N_vis, 4096]
vision_mask:     bool [B, N_vis]
vision_time:     float32 [B, N_vis]   # timestamp of source frame, seconds
vision_cache_id: int64 [B]            # increment on FAST_SENSOR_REFRESH
```

`VisualCache` stores `hpt_vision`, `vision_mask`, `vision_time`. Fast ticks consume cache without calling `VisionEncoder`.

### 3.3 Audio (sensor refresh)

**LEDGER-LOCKED:** 8 layers `d=768`, SwiGLU 2048, frontend `128×768` ⇒ mel bins **128**.

```text
audio_mel:   bf16 [B, T_audio, 128]
audio_mask:  bool [B, T_audio]
hpt_audio:   bf16 [B, T_audio, 4096]
```

### 3.4 NCES

**LEDGER-LOCKED:** encoder input feature width **128**; 6 layers `d=2048`; out `2048→4096`.

**FROZEN node classes:** ROOT, HEAD, TORSO, L_ARM, R_ARM, L_HAND, R_HAND, L_LEG, R_LEG, optional extras.

**IMPLEMENTATION CONTRACT — node feature packing `D_NCES_IN=128`:**

| Slice | Dims | Semantics |
|---|---|---|
| 0:3 | 3 | position (m), gravity-aligned world or body-relative as flagged |
| 3:9 | 6 | rot6d (first two rotation-matrix rows) |
| 9:12 | 3 | linear velocity |
| 12:15 | 3 | angular velocity |
| 15:16 | 1 | node valid {0,1} |
| 16:18 | 2 | abstract grasp {aperture, squeeze} (hands; else 0 + valid=0) |
| 18:19 | 1 | contact_binary |
| 19:22 | 3 | optional wrench (0 if absent) |
| 22:25 | 3 | ROOT-only: gravity dir (unit) |
| 25:27 | 2 | ROOT-only: support flags |
| 27:30 | 3 | ROOT-only: momentum summary |
| 30:32 | 2 | frame flags: coordsys {world=1,body=0}, wrench_present |
| 32:128 | 96 | reserved / topology one-hot / pad; unused dims are 0 and must not imply validity |

**OPEN DECISION:** `N_nodes_max`. Freeze hypothesis 24–40. Ledger does not depend on `N_nodes` (transformer over padded nodes). **IMPLEMENTATION CONTRACT pending approval:** `N_NODES_MAX = 40`.

```text
nces_feat:     bf16 [B, N_nodes, 128]
nces_valid:    bool [B, N_nodes]          # missing nodes False; NOT valid zeros
nces_type:     int32 [B, N_nodes]         # enum NodeType
contact_idx:   int32 [B, E_max, 2]        # node_i, node_j_or_object
contact_attr:  bf16 [B, E_max, 8]         # normal3, mag_bin, conf, pad
contact_valid: bool [B, E_max]
hpt_nces:      bf16 [B, N_nodes, 4096]
```

Tactile **LEDGER-LOCKED:** `256 → 4096` plus `4096×4096` mix.

```text
tactile:       bf16 [B, 256]
tactile_valid: bool [B]
hpt_tactile:   bf16 [B, 1, 4096]          # mixed into PHYSICAL stream
```

### 3.5 ECD

**LEDGER-LOCKED:** MLP `128→1024→4096`; topology `32×32→512`; 3 layers `d=512` SwiGLU 1408; `512→4096`.

**IMPLEMENTATION CONTRACT — ECD vector `128` (capability only, not controller internals):**

| Slice | Dims | Content |
|---|---|---|
| 0:16 | 16 | topology class one-hot / hash bins |
| 16:32 | 16 | effector presence bitmask (hands, feet, gripper, mobile base, …) |
| 32:48 | 16 | workspace AABB / reach class (normalized) |
| 48:56 | 8 | locomotion modes bitmask |
| 56:64 | 8 | manipulation modes bitmask |
| 64:80 | 16 | sensor availability bitmask |
| 80:88 | 8 | payload class |
| 88:96 | 8 | dexterity class |
| 96:112 | 16 | mobility constraints |
| 112:128 | 16 | reserved |

```text
ecd_raw:    float32 [B, 128]
ecd_topo:   float32 [B, 32, 32]           # adjacency / type grid
hpt_ecd:    bf16 [B, 1, 4096]             # PHYSICAL or dedicated prefix token
```

Do not put gear ratios, gains, or PWM into `ecd_raw`.

### 3.6 Language

**FROZEN:** vocab 65536, untied `wte` and `lm_head`.

```text
input_ids:     int32 [B, T_lang]
lang_mask:     bool [B, T_lang]
hpt_lang:      bf16 [B, T_lang, 4096]
lm_logits:     fp32 [B, T_lang, 65536]    # only if language emitted
null_instruction: input_ids may be a single NULL token
```

Special tokens **IMPLEMENTATION CONTRACT** (must be inside 65536, not extra params):

```text
<|am_pad|> <|am_bos|> <|am_eos|> <|am_null_instruction|>
<|am_mod_vision|> <|am_mod_language|> <|am_mod_physical|> <|am_mod_agency|>
<|am_act|> <|am_observe|> <|am_hold|> <|am_wait|>
<|am_allow|> <|am_defer|> <|am_block|>
<|am_mem_write|> <|am_mem_read|>
```

Count of specials ≤ 16 **LEDGER-LOCKED** (`16 * d` special embeddings). The freeze also has 16 modality embeddings separate from vocab. Do not add a 17th special embedding table.

### 3.7 HPT sequence

Packed token stream per batch item:

```text
hidden_states:  bf16 [B, S, 4096]
modality_ids:   int8 [B, S]
attention_mask: bool [B, S]               # True = keep
token_time:     float32 [B, S]
position_ids:   int32 [B, S]              # RoPE index in working window
layer_kind:     # implicit by module: Fast 0–11, Slow 12–39
moe_token_mask: bool [B, S]               # True iff PHYSICAL and layer in L20–L39
```

**FROZEN working window:** 4096–8192. **OPEN DECISION:** train `S_max`. **IMPLEMENTATION CONTRACT pending:** `S_max = 4096` for Phase II kernels (8192 is freeze-legal; do not ship both as “the” 32B identity).

**Attention semantics (IMPLEMENTATION CONTRACT pending approval):**

- Shared GQA over the packed sequence.
- Fast layers: tokens present = VISION (cached) + PHYSICAL (+ ECD as physical). No LANGUAGE/AGENCY FFN.
- Slow layers: all four modalities.
- Causal along `token_time` for language and agency; bidirectional within the same timestamp for vision/NCES **same-tick** tokens. **This mixed mask is not fully specified in the freeze.** See Section 10 OPEN DECISIONS.
- Hybrid 3 sliding-window : 1 full. Window size **HYPOTHESIS 4096** in freeze. If `S_max=4096`, window equals full sequence on Fast/Slow working window.

**FlashAttention / SDPA:** reference math is `GQA + QK-RMSNorm(shared, dim=128) + mask + RoPE`. SDPA is the correctness oracle. FlashAttention-2/3 is allowed **only** if QK-norm is applied before FA, GQA grouping matches `32/8/128`, and the mask (causal/window/padding) matches SDPA within bf16. Do not drop QK-norm to get a kernel.

### 3.8 Z hybrid state

| Field | Kind | Shape | Supervision |
|---|---|---|---|
| `Z_self` | structured | `[B, N_nodes, 4096]` or pooled `[B, 1, 4096]` from NCES | NCES losses |
| `Z_contact` | structured sparse | `[B, E_max, 4096]` + `contact_valid` | `L_contact` if labeled |
| `Z_uncertainty` | structured scalars | `[B, 9]` aux heads also serve here | `L_gate` proxies |
| `Z_objects` | weakly structured slots | `[B, N_obj, 4096]`, `obj_mask` | optional IDs; permanence via slot tracking |
| `Z_humans` | weakly structured slots | `[B, N_hum, 4096]`, `hum_mask` | optional |
| `Z_scene` | fully latent | `[B, N_scene, 4096]` | JEPA `L_future` only |
| `Z_dynamics` | fully latent | `[B, N_dyn, 4096]` | JEPA only |

**OPEN DECISION:** `N_obj, N_hum, N_scene, N_dyn`. Ledger does not include extra slot embeddings beyond counted modules. Slots are **views / pooled tokens from HPT**, not new billion-scale tables.

**IMPLEMENTATION CONTRACT pending:** `N_obj=16`, `N_hum=4`, `N_scene=32`, `N_dyn=16` as **pooling queries** (parameters already inside EAC/HPT), not new matrices.

```text
Z: packed bf16 [B, N_Z, 4096]
Z_kind: int8 [B, N_Z]
Z_mask: bool [B, N_Z]
```

### 3.9 Memory

**LEDGER-LOCKED working:** 256 learned slots `[256, 4096]`; one GQA cross-attn (`P_attn`); write `d×d+d`.

```text
wm_slots:     bf16 [B, 256, 4096]         # persistent across Fast ticks
wm_write_gate: bf16 [B, 256]
```

**LEDGER-LOCKED episodic:** 2 × VE-width layers + `3 * 2048 * 4096` projections.

**Bounded store (IMPLEMENTATION CONTRACT pending):** ring of `N_epi = 1024` compressed events per episode max, each `2048-d`, **not** an unbounded Python list. Overflow: evict oldest. Train-time differentiable path uses the compressor on the current write only; the ring is detached except the write/read projections.

**LEDGER-LOCKED semantic:** `Q: d×d`, `K,V: d×512`.

**Bounded store pending:** `N_sem = 4096` keys. Retrieve top-`k_sem=8`. Evict lowest retrieval count.

Gated write: surprise / contact / intent-switch / speech. No write every Fast tick.

### 3.10 EAC / GCIS

**LEDGER-LOCKED:** 11 intent query embeddings (`11 * d`) ⇒ **K_act = 8** ACT candidates + OBSERVE + HOLD + WAIT. 64 agency tokens. 4 GCIS layers. Qθ MLP `(4d)→d→1`. Gate `(d+32)→1024→3`. 9 aux heads `d→512→1`.

```text
agency_tokens:   bf16 [B, 64, 4096]
intent_latents:  bf16 [B, 11, 4096]
intent_mask:     bool [B, 11]             # all True unless padded
intent_kind:     int8 [B, 11]             # 0–7 ACT, 8 OBSERVE, 9 HOLD, 10 WAIT
intent_scores:   fp32 [B, 11]             # Qθ
gate_logits:     fp32 [B, 11, 3]          # ALLOW, DEFER, BLOCK
aux_heads:       fp32 [B, 9]              # order frozen below
G_t:             bf16 [B, T_lang, 4096] or NULL token
```

Aux order **FROZEN names:** uncertainty, novelty, social_relevance, env_change, intervention_value, action_cost, physical_risk, persistence, inhibition.

`Qθ` input concat: `[I_k; Z_pool; A_pool; G_pool]` each `4096` → `16384`. ECD `E_t` enters via `hpt_ecd` into `Z_pool` / HPT, not a fifth concat (would add params). **This matches ledger.** Do not add a fifth projection.

### 3.11 Flow NCES

**LEDGER-LOCKED:** 12 layers `d=2048`; AdaLN from 256-d time; IO `512↔2048`.

```text
nces_traj:     bf16 [B, H_chunk, N_nodes, D_node]
nces_traj_valid: bool [B, H_chunk, N_nodes]
nces_packed:   bf16 [B, H_chunk, 512]     # pack/unpack; 512-d interface FROZEN
flow_t:        fp32 [B]                   # t in [0,1]
flow_pred:     bf16 [B, H_chunk, 512]     # vector field
```

**OPEN DECISION:** `H_chunk` in {16,32} and `D_node` packing into 512. **IMPLEMENTATION CONTRACT pending:** `H_chunk=16`; pack per-step as MLP over masked nodes into 512 (weights are the frozen `512*2048` IO, not a new giant table).

HOLD uses the same tensor as a **desired-state trajectory** (repeat current NCES or explicit hold target). Never encode HOLD as all-zero torque.

NFE is config, not a claimed empirical optimum. Default inference policy keys: `nfe_fast`, `nfe_precision` (unset until measured).

### 3.12 Adapter boundary (outside 31.740B)

```text
nces_desired:  bf16 [B, H_chunk, N_nodes, D_node]
ecd_raw:       float32 [B, 128]
robot_state:   bf16 [B, D_robot]          # family-specific, not HPT dim
adapter_out:   bf16 [B, H_chunk, D_cmd]   # joint / EE / impedance setpoints
```

`D_robot`, `D_cmd` are per family. Not in freeze total.

---

## 4. Immutable AMATERASU-32B config

Identity string: `AMATERASU-32B-v0.1`. Changing any field below **forfeits** the name. `validate_freeze` hashes this JSON.

```text
name: AMATERASU-32B-v0.1
d_model: 4096
n_heads: 32
n_kv_heads: 8
d_head: 128
d_ff: 11008
d_ff_expert: 4096
n_routed_experts: 8
n_shared_experts: 1
moe_topk: 2
n_fast_layers: 12          # L0–L11
n_slow_layers: 28          # L12–L39
n_moe_layers: 20           # L20–L39
n_slow_dense_physical: 8   # L12–L19
vocab_size: 65536
untied_lm_head: true
qk_norm_shared: true       # P_qk = 256
bias: false
n_modality_emb: 16
n_special_emb: 16
ve_d: 2048
ve_layers: 36
ve_heads: 16
ve_kv: 4
ve_d_head: 128
ve_d_ff: 5504
tubelet: [2, 14, 14, 3]
nces_layers: 6
nces_d: 2048
nces_in: 128
audio_d: 768
audio_layers: 8
audio_d_ff: 2048
audio_mel: 128
tactile_in: 256
ssm_layers: 4
ssm_d_inner: 8192
ssm_d_state: 128
ssm_dt_rank: 256
ssm_d_conv: 4
wm_slots: 256
dyn_layers: 8
dyn_d: 3072
dyn_heads: 24
dyn_kv: 6
dyn_d_ff: 8192
eac_gcis_layers: 4
eac_queries: 11
eac_agency_tokens: 64
eac_aux_heads: 9
flow_layers: 12
flow_d: 2048
flow_io: 512
ecd_in: 128
frozen_total: 31740290560
```

Training JSON may change batch, stage, mixture, NFE, `S_max` (within freeze-legal range). It **must not** change the block above.

---

## 5. Exact parameter-accounting map

`amaterasu.model.accounting.account(model)` walks **named parameters** with `requires_grad=True`, excludes `adapter.*`, and compares to this table. Inequality → **raise SystemExit / hard fail** with per-component diff. No rounding.

**TOTAL FROZEN: 31,740,290,560**

| Component | Stored | Notes |
|---|---:|---|
| Vision encoder | 1,605,837,824 | FAST_SENSOR_REFRESH |
| Embeddings | 537,001,984 | wte+lm_head+16 mod+16 special |
| Shared HPT attn+norm | 1,677,895,680 | 40 × (P_attn + d + 256) |
| Vision FFNs | 5,410,816,000 | 40 × (P_FFN + d) |
| Language FFNs | 3,787,571,200 | 28 × (P_FFN + d) |
| Physical dense FFNs | 2,705,408,000 | 20 × (P_FFN + d) |
| Physical experts | 9,060,433,920 | 20 × (P_moe_stored + d) |
| Agency FFNs | 3,787,571,200 | 28 × (P_FFN + d) |
| NCES encoder | 274,490,880 | |
| Audio encoder | 59,879,424 | |
| Tactile | 17,825,792 | |
| SSM | 428,032,000 | |
| Memory | 194,523,648 | WM+epi+sem |
| Latent dynamics | 830,523,392 | |
| EAC/GCIS | 799,404,544 | |
| Flow | 546,491,392 | |
| ECD | 16,583,680 | |
| **TOTAL** | **31,740,290,560** | |

**Execution-path active (must match freeze graphs):**

| Graph | Params | Includes |
|---|---:|---|
| TOTAL | 31,740,290,560 | all universal |
| SLOW_ACTIVE | 24,885,565,952 | no lm_head, no flow, Physical **top-2+shared** not 8 stored |
| SLOW_ACT_HOLD | 25,432,057,344 | SLOW_ACTIVE + flow |
| FAST_STATE_ACTIVE | 4,546,563,584 | no VE, no audio, no Slow, no MoE bank, no dyn/EAC/flow |
| FAST_SENSOR_REFRESH_ACTIVE | 1,605,837,824 | vision; +audio 1,665,717,248 if audio refresh |
| FAST_ALWAYS coincident | 6,212,280,832 | FAST_STATE + VE + audio |
| FAST_ACT_HOLD coincident | 6,758,772,224 | FAST_ALWAYS + flow |
| FAST_STATE + flow cached | 5,093,054,976 | HOLD/ACT without visual encode |

Physical MoE **stored** vs **routed active** per MoE layer:

```text
stored = 453,017,600 + 4096   # experts+router+Physical RMS
active = 151,027,712 + 4096   # 3 * P_exp + router + RMS
```

LM head **268,435,456** is in TOTAL, not in SLOW_ACTIVE.

Adapters: reported separately; **must not** appear in TOTAL.

---

## 6. Distributed topology

**Primary stack:** PyTorch + **Megatron-Core-style** TP/PP/EP (custom AMATERASU modules, not a pretrained Megatron GPT/VLM class) + **Transformer Engine** GEMMs where they preserve exact GQA/SwiGLU math + **FlashAttention** only under Section 3.7. Optimizer: AdamW, fp32 master weights, bf16 grads. FSDP2/ZeRO-3 only for non-expert replicas if EP already shards experts.

Do **not** import Hugging Face `transformers` model classes as the training core.

**Physical MoE EP:** `E=8` routed experts ⇒ `EP ≤ 8`. Shared expert replicated or EP-rank-0. **Actual sparse dispatch** (indices → all-to-all → expert GEMM → reverse). No “run all 8 and mask.”

| GPU | Suggested start (IMPLEMENTATION CONTRACT) | Precision |
|---|---|---|
| H100 80GB | TP=8, PP=8, EP=8, DP=world/(TP·PP·EP), activation ckpt on Slow | bf16, fp32 router |
| H200 141GB | TP=4, PP=8, EP=8, DP=remainder | bf16; FP8 TE after proxy |
| B200-class | TP=2–4, PP=4–8, EP=8 | bf16 first; FP8 after loss match |

Context/sequence parallel (`CP`) only if `S_max=8192` and memory requires it.

Init and accounting scripts must run on **one process** with meta/CPU tensors to verify 31,740,290,560 **before** multi-GPU train.

---

## 7. Checkpoint architecture

Format: **safetensors only** for tensors. At `init.py`: construct modules → in-scratch init → **immediate** `save_model` safetensors (sharded by TP/PP/EP rank). No pickle of weights.

Shard naming: `rank{tp}-{pp}-{ep}.safetensors`. Manifest JSON:

```text
format: amaterasu-ckpt-v1
model_id: AMATERASU-32B-v0.1
freeze_hash: ...
frozen_total: 31740290560
global_step, stage, mixture_id
rng: python/numpy/torch/cuda
mesh: tp pp ep dp world_size
optim: sharded state (safetensors or dist-opt format, not pickle weights)
scheduler, curriculum, dataset_sampler
router_ema / expert bias if used
```

Resume with different world size: allowed only if TP/PP/EP factors remain compatible with layer/expert sharding; otherwise fail loud.

---

## 8. Runtime execution graphs

One module, **explicit modes**. No single `forward()` that always runs VE+EAC+Flow.

| Mode | Modules | Flow | Vision encode |
|---|---|---|---|
| `TRAIN` | stage-dependent; losses masked | if `L_action` active | if visual tokens in batch |
| `SLOW_AGENCY` | Slow HPT, mem, dyn, EAC, gate | no | cache or refresh if new |
| `FAST_OBSERVE` | FAST_STATE | off | cache |
| `FAST_WAIT` | FAST_STATE + deferred intent metadata | off | cache |
| `FAST_HOLD` | FAST_STATE + Flow desired-state | on | cache |
| `FAST_ACT` | FAST_STATE + Flow | on | cache |
| `FAST_SENSOR_REFRESH` | VisionEncoder ± Audio | n/a | on, then update cache |

SSM state persists across Fast modes. Detach policy **IMPLEMENTATION CONTRACT pending:** truncated BPTT `T_unroll=16` Fast ticks; detach SSM state beyond window. Reset on `episode_reset=True`.

---

## 9. File-by-file implementation order

Phase II (after this blueprint is approved):

1. `constants.py`, `Amaterasu32BConfig`, freeze JSON + hash  
2. `tensors/modality.py`, `dtypes.py`  
3. `rmsnorm.py`, `qk_norm.py`  
4. `gqa.py`, `rope.py`, `mask.py`  
5. `swiglu.py`  
6. `hpt/dispatch.py`, `fast_layer.py`  
7. `slow_layer.py`  
8. `router.py`, `physical_moe.py` (true sparse)  
9. `language/embeddings.py`  
10. `hpt/stack.py`  
11. Remaining counted modules as empty **parameter-correct** shells only if Phase II is defined as HPT+embeddings first — **NO.** Phase II must assemble **full** `Amaterasu32B` parameter set (vision, NCES, SSM, memory, dyn, EAC, flow, ECD, tactile, audio) so accounting returns **31,740,290,560** before Phase III behavior is complete.

**Phase II completeness rule:** every universal parameter tensor must exist and be initialized. Forward may be mode-limited, but `account(model).total == 31740290560` is the gate.

Then Phase III fills real vision/NCES/SSM/memory/Z compute (already instantiated).  
Phase IV fills dynamics/EAC/flow/adapter behavior.  
Phase V data/losses.  
Phase VI distributed loop.  
Phase VII clocks.

Scripts: `scripts/init_and_account.py` is the Phase II exit criterion.

---

## 10. Missing architectural decisions (STOP)

These are **not** implemented until approved. They are not an invitation to redesign HPT/MoE/32B.

1. **Vision geometry** `N_cam, T_video, H, W`. Freeze only tubelet and encoder width/depth.  
2. **Mixed attention mask** (causal language vs bidirectional same-tick vision/NCES). Freeze says RoPE + hybrid window; not the cross-modality causal pattern.  
3. **`S_max` 4096 vs 8192** for the first training kernel.  
4. **`N_NODES_MAX`** (proposed 40).  
5. **`H_chunk` 16 vs 32** and exact pack `N_nodes × D_node → 512`.  
6. **Z slot counts** `N_obj, N_hum, N_scene, N_dyn` if treated as extra parameters — extra embeddings would **break TOTAL**. Must remain pooling views.  
7. **Episodic/semantic ring sizes** (proposed 1024 / 4096) — bounded compute, not new weight tables.  
8. **SSM TBPTT length** (proposed 16).  
9. **Tokenizer training corpus** (from-scratch BPE/Unigram; vocab 65536 frozen). Corpus mix is data-config, but must not add tokens beyond 65536.  
10. **Distributed first cluster** (H100 vs H200 vs B200) — topologies listed; one must be chosen for Phase VI, not for Phase II accounting.

No `ARCHITECTURE IMPLEMENTATION CONFLICT` with the ledger is known **if** Z slots and memory rings add **zero** new parameters and vision/NCES/flow IO match the frozen Linear shapes.

If Phase II accounting ≠ `31,740,290,560`, that is a **conflict**: fix the implementation, not the freeze number.

---

## 11. Phase I gate

Stop here for consistency verification.

Do not generate the codebase until this blueprint is approved, including the OPEN DECISIONS in Section 10 or an explicit instruction to use the listed IMPLEMENTATION CONTRACT defaults.

Author: **MagistrTheOne | NULLXES**
