# NULLXES AMATERASU

From-scratch **Embodied Agency Foundation Model** (EAFM). Not a fine-tuned VLM, not an LLM with a robot head, not π0 / GR00T / OpenVLA / Qwen / Llama weights.

**Author:** MagistrTheOne | NULLXES  
**AMATERASU-32B v0.1:** architecture frozen · **31,740,290,560** universal parameters  
**AMATERASU-72B:** approved scale spec only (`71,689,455,872`)

The job is to remain a physical agent when `instruction = NULL`: observe → persistent state → candidate intent → predicted consequence → **ACT** or **OBSERVE / HOLD / WAIT**.

## Specs

| Document | Role |
| --- | --- |
| [AMATERASU_EAFM_ARCHITECTURE_v0.1.md](AMATERASU_EAFM_ARCHITECTURE_v0.1.md) | Frozen architecture |
| [AMATERASU_32B_IMPLEMENTATION_BLUEPRINT.md](AMATERASU_32B_IMPLEMENTATION_BLUEPRINT.md) | Implementation map |
| [AMATERASU_DATASET_REGISTRY_v0.1.md](AMATERASU_DATASET_REGISTRY_v0.1.md) | Which sources may emit `AMATERASUSample` |

Checkpoints are **native safetensors** (`amaterasu-ckpt-v1`), not Hugging Face `transformers`. Identity lives in `manifest.json` + `configs/model/amaterasu_32b_v0.1.json`. Init shards (`checkpoints/amaterasu_32b_v0.1_init/`, ~127 GB fp32 on disk) stay **local** — they are not in git.

## Data plane

We do not hand-build one giant “AMATERASU dataset”. Open corpora (HiFi, DROID, OXE, BONES, G1, …) pass **source converters** → `AMATERASUSample` → research/commercial mixers → 9-stage curriculum.

Registry v0.1 is **29 families**, not Hub spam. Two axes:

- `TECHNICAL_ADMISSION`: FOUNDATION / QA / REJECT  
- `LEGAL_ADMISSION`: COMMERCIAL / RESEARCH_ONLY / CONDITIONAL / UNRESOLVED  

Provenance is a graph. A G1 set derived from BONES stays research-only even if a mirror says Apache-2.0. Two rows exist **now** with `scale = 0` (red holes): **Agency-NULLXES** and **Simulation-NULLXES**.

Mixture `32/28/18/12/10` is a commercial *manipulation profile*, not the eternal diet. Stage weights live in `configs/train/` and `configs/data/`.

## Next (locked)

Registry v0.1 → coverage matrix (in the registry) → **Agency Dataset Spec** → **Dynamics/Sim Dataset Spec** → OXE child license manifest → converter contracts → converters.

Do not start converters before the two NULLXES specs. Imitation-only samples are not EAFM.

## Layout

```text
amaterasu/     model, tensors, data adapters, train, infer
configs/       freeze JSON, stages 1–9, mixtures, cluster topologies
scripts/       init_and_account, train, resume, infer_runtime
checkpoints/   local safetensors only (gitignored)
```
