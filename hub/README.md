---
license: apache-2.0
library_name: amaterasu
tags:
  - embodied-ai
  - physical-ai
  - robotics
  - safetensors
  - amaterasu
  - nullxes
pipeline_tag: robotics
---

# NULLXES AMATERASU-32B v0.1

**Author:** MagistrTheOne | NULLXES  
**Hub:** [MagistrTheOne/NULLXES-AMATERASU](https://huggingface.co/MagistrTheOne/NULLXES-AMATERASU)  
**Code:** [github.com/MagistrTheOne/NULLXES-AMATERASU](https://github.com/MagistrTheOne/NULLXES-AMATERASU)

From-scratch Embodied Agency Foundation Model. Native format `amaterasu-ckpt-v1`. Not Transformers, not π0 / GR00T / OpenVLA / Qwen / Llama weights. Do not `AutoModel.from_pretrained`.

## Identity

| | |
|---|---|
| model_id | `AMATERASU-32B-v0.1` |
| format | `amaterasu-ckpt-v1` |
| frozen_total | `31,740,290,560` |
| freeze_hash | `c1ff97b33d3f9280ccd5e066306a3ade4a23d8d7bbec484a3faa523a476129c2` |
| init dtype | fp32 shards |
| train dtype | bf16 |

## Architecture

| | |
|---|---|
| d_model | 4096 |
| attention | GQA 32 / 8 / 128 |
| FFN | SwiGLU `d_ff=11008` |
| HPT | 40 layers: Fast 12, Slow dense 8, Physical MoE 20 |
| MoE | 8 routed + 1 shared, top-2 |
| vocab | 65536, untied `wte` + `lm_head` |
| vision | 36-layer encoder, train `224×224`, `T_CLIP=16`, `N_CAM_MAX=6` |
| NCES | 6 layers, 128-d in, `N_NODES_MAX=64` |
| clocks | Slow ~2–5 Hz, Fast ~30–100 Hz |

## Parameter ledger

| component | params |
|---|---:|
| vision_encoder | 1,605,837,824 |
| embeddings | 537,001,984 |
| shared_hpt_attn_norm | 1,677,895,680 |
| vision_ffns | 5,410,816,000 |
| language_ffns | 3,787,571,200 |
| physical_dense_ffns | 2,705,408,000 |
| physical_experts | 9,060,433,920 |
| agency_ffns | 3,787,571,200 |
| nces_encoder | 274,490,880 |
| audio_encoder | 59,879,424 |
| tactile | 17,825,792 |
| ssm | 428,032,000 |
| memory | 194,523,648 |
| latent_dynamics | 830,523,392 |
| eac_gcis | 799,404,544 |
| flow | 546,491,392 |
| ecd | 16,583,680 |
| **TOTAL** | **31,740,290,560** |

## Files in this repo

| file | what |
|---|---|
| `README.md` | this card |
| `amaterasu_32b_v0.1.json` | freeze config |
| `nces-circuit0.safetensors` | Stage-1 Circuit-0 NCES encoder after HiFi parquet train (trainable subset only) |
| `metrics.json` | circuit-0 run metrics |

Full 32B init (`51 × .safetensors` + `manifest.json`, ~119 GB fp32) is **not** this Hub folder. Load universal weights from the volume checkpoint `amaterasu_32b_v0.1_init`, then overlay NCES from `nces-circuit0.safetensors`.

## Circuit-0 (this release)

- Data: [HiFi-UMI-2K](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K) parquet only (CC BY 4.0). No video. No `intent_label` (hand motion ≠ ACT).
- Trainable: `nces` only (`274,490,880`). Rest of freeze graph frozen.
- Hardware: 1× NVIDIA H200 141 GB, bf16 autocast. Not full 32B AdamW.
- Gate: finite `L_mm`, checkpoint write, resume from `amaterasu-ckpt-v1`.

## Load

Use AMATERASU `resume_modules` for the 32B init shards, then `nces.load_state_dict` from `nces-circuit0.safetensors`.
