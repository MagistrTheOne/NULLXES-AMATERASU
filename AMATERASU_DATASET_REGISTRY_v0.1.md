# AMATERASU Dataset Registry v0.1

**Author:** MagistrTheOne | NULLXES  
**Audit date:** 28 August 2026  
**Status:** source-of-truth for which corpora may produce `AMATERASUSample`  
**Not:** converters, downloaders, or a merged “AMATERASU dataset”

Architecture already defines the sample contract (`AMATERASUSample`). This registry decides **who is allowed to emit it**. Hugging Face mirrors are indexes, not provenance.

Claim tags: **VERIFIED** (primary card/paper/license page this audit), **ESTIMATE**, **UNVERIFIED**.

---

## 0. How to read this document

Rows are **dataset families**, not individual Hub repos. Forty `G1_pick_*` dumps are one family. Two admission axes are independent:

| Axis | Values | Meaning |
| --- | --- | --- |
| `TECHNICAL_ADMISSION` | `FOUNDATION` / `QA` / `REJECT` | physical signal, NCES/ECD convertibility, scale, uniqueness |
| `LEGAL_ADMISSION` | `COMMERCIAL` / `RESEARCH_ONLY` / `CONDITIONAL` / `UNRESOLVED` | provenance graph, not the mixture JSON filename |

BONES-SEED is `FOUNDATION` + `RESEARCH_ONLY`. That is admission, not rejection.

`commercial_safe` is derived from `LEGAL_ADMISSION` and `derived_from`. If `C ← B ← BONES`, Apache-2.0 on C does **not** wash the parent. Those derivatives stay `CONDITIONAL` until the chain is licensed independently.

### Admission gate

```text
SOURCE
  │
  ├─ Original provenance known? ──NO──> REJECT
  │
  ├─ License resolved? ───────────NO──> LEGAL = UNRESOLVED (quarantine)
  │
  ├─ Physical signal useful? ─────NO──> REJECT
  │
  ├─ NCES convertible? ───────────NO──> QA / later
  │
  ├─ Duplicate information? ──────YES─> lower priority (keep if unique residual)
  │
  ├─ Meaningful scale? ───────────NO──> QA
  │
  └───────────────────────────────────> FOUNDATION
```

`NCES_score` / `ECD_score` are 0–5 **ESTIMATE** of convertibility into canonical nodes / embodiment cards, not dataset quality.

Stages map to `configs/train/stage_01`–`stage_09`.

---

## 1. Master table (29 families)

| # | dataset_family | TECH | LEGAL | purpose | embodiment | NCES | ECD | stage | P | unique_information |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | `hifi-umi-2k` | FOUNDATION | COMMERCIAL | manipulation | HUMAN | 5 | 3 | 1–3 | P0 | large-scale human bimanual manipulation, 6-cam, robot-free |
| 2 | `droid` | FOUNDATION | CONDITIONAL | manipulation | FRANKA | 4 | 4 | 2–5 | P0 | diverse real Franka manipulation + language |
| 3 | `oxe` | FOUNDATION | UNRESOLVED | manipulation | MULTI | 5 | 4 | 4–5 | P0 | cross-embodiment robot action (per-subset license) |
| 4 | `bones-seed` | FOUNDATION | RESEARCH_ONLY | motion | HUMAN | 5 | 4 | 2–4 | P0 | human whole-body motion prior (SOMA + G1 CSV) |
| 5 | `agency-nullxes` | FOUNDATION | COMMERCIAL | agency | MULTI | 2 | 2 | 7–8 | P0 | intervention / deliberate non-intervention (**scale = 0**) |
| 6 | `simulation-nullxes` | FOUNDATION | COMMERCIAL | dynamics | SIM | 4 | 4 | 6, 9 | P0 | branched counterfactual futures (**scale = 0**) |
| 7 | `humanoid-everyday` | FOUNDATION | COMMERCIAL | manipulation | G1 | 4 | 4 | 4–5 | P1 | open-world G1 teleop: loco + dexterous + human interaction |
| 8 | `unifolm-wbt` | FOUNDATION | CONDITIONAL | motion | G1 | 5 | 5 | 2–5 | P1 | real G1 whole-body teleoperation (legs+torso+hands) |
| 9 | `unitree-g1-dex-teleop` | FOUNDATION | COMMERCIAL | manipulation | G1 | 4 | 4 | 3–5 | P1 | official Unitree G1 Dex1/Dex3 task collection |
| 10 | `mosaic-g1` | FOUNDATION | CONDITIONAL | motion | HUMAN→G1 | 5 | 4 | 4 | P1 | paired human → humanoid retargeting |
| 11 | `deco-50` | FOUNDATION | COMMERCIAL | contact | MULTI | 3 | 3 | 3, 6 | P1 | tactile bimanual interaction |
| 12 | `robomind` | FOUNDATION | CONDITIONAL | manipulation | MULTI | 4 | 4 | 4–5 | P1 | multi-embodiment real teleop at ImageNet-ish robot scale |
| 13 | `bridge-v2` | FOUNDATION | CONDITIONAL | manipulation | WIDOWX | 4 | 3 | 4–5 | P2 | WidowX kitchen-ish; overlaps OXE — residual if OXE subset drops it |
| 14 | `agibot-world` | FOUNDATION | RESEARCH_ONLY | manipulation | MULTI | 4 | 4 | 3–5 | P1 | large dual-arm real scenes; NC-SA |
| 15 | `ego4d` | FOUNDATION | RESEARCH_ONLY | observation | HUMAN | 1 | 1 | 1 | P1 | in-the-wild egocentric physical observation |
| 16 | `epic-kitchens` | FOUNDATION | RESEARCH_ONLY | observation | HUMAN | 1 | 1 | 1 | P2 | kitchen egocentric; NC |
| 17 | `robocoin-g1` | FOUNDATION | COMMERCIAL | manipulation | G1 | 4 | 4 | 3–5 | P2 | G1edu task family (hours, not 50-ep toys) |
| 18 | `sonic-g1-tracking` | QA | RESEARCH_ONLY | dynamics | G1 | 4 | 4 | 6, 9 | P2 | SONIC policy rollouts on BONES-retargeted G1 (**derived**) |
| 19 | `amass-smpl` | QA | CONDITIONAL | motion | HUMAN | 4 | 2 | 2 | P3 | SMPL mocap prior; largely dominated by BONES if BONES is in |
| 20 | `artvip` | QA | COMMERCIAL | dynamics | SIM | 2 | 3 | 6, 9 | P2 | articulated object digital twins for contact/sim |
| 21 | `libero` | QA | CONDITIONAL | manipulation | FRANKA | 3 | 3 | 3 | P3 | tabletop language-conditioned; OXE overlap |
| 22 | `openeai-aggregate` | QA | UNRESOLVED | schema | MULTI | — | — | — | P2 | **schema reference only** — do not feed AMATERASU |
| 23 | `contact-tactile-open` | QA | UNRESOLVED | contact | MULTI | 2 | 2 | 3 | P2 | leftover tactile/HOI after DECO; needs per-set audit |
| 24 | `ego-hand-micro` | QA | UNRESOLVED | manipulation | HUMAN | 4 | 2 | — | P2 | tiny ego-hand LeRobot (converter gold, not 32B fuel) |
| 25 | `ikea-g1-challenge` | QA | COMMERCIAL | manipulation | G1 | 3 | 3 | — | P3 | long-horizon assembly; tiny N |
| 26 | `community-g1-lerobot` | REJECT | UNRESOLVED | manipulation | G1 | 3 | 3 | — | P3 | single-task 50–100 ep Hub spam |
| 27 | `habitat-humanoids` | REJECT | RESEARCH_ONLY | motion | SIM | 2 | 2 | — | P3 | SMPL-X avatars, NC; not NCES action |
| 28 | `g1-hardware-cad` | REJECT | COMMERCIAL | — | G1 | 0 | 0 | — | P3 | printable parts, not trajectories |
| 29 | `egodex` | QA | RESEARCH_ONLY | manipulation | HUMAN | 3 | 2 | 1–2 | P3 | ego dexterous; commercial mix already drops it |

Red holes that must stay visible: **#5 and #6 have `scale = 0`.**

---

## 2. Family cards

License verification date for all cards unless noted: **2026-08-28**.

### 2.1 `hifi-umi-2k`

| field | value |
| --- | --- |
| source_org | Simple AI / simple-world-lab |
| primary_source | [simple-world-lab/HiFi-UMI-2K](https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K), paper [arXiv:2607.25895](https://arxiv.org/abs/2607.25895) |
| bucket / legal | FOUNDATION / COMMERCIAL |
| purpose | manipulation (human, robot-free) |
| embodiment | HUMAN (bimanual EE + gripper) |
| scale | **VERIFIED** 2,000 h public; 6 cams; 480+ scenes; ~192.3M rows on Hub card; ~16 TB **ESTIMATE** from card chatter — treat storage as Hub-reported |
| video | 6 synchronized views; faces masked |
| proprio | calibrated bimanual trajectories, 3 mm **VERIFIED** (paper) |
| actions | human EE xyz+rot6d+gripper → NCES pack |
| tactile_contact | none dedicated |
| language | annotations + subtask boundaries |
| NCES / ECD | 5 / 3 |
| stage | 1–3 (representation → motion → manipulation) |
| license | CC BY 4.0 **VERIFIED** (card + paper) |
| commercial_safe | YES |
| license_verified | 2026-08-28, HF card + arXiv HTML |
| derived_from | original capture |
| priority | P0 |
| unique_information | large-scale human bimanual manipulation |
| notes | Anchor of human physical interaction. Not Franka joints. Adapter already sketched in `amaterasu/data/adapters/hifi_umi.py`. |

### 2.2 `droid`

| field | value |
| --- | --- |
| source_org | DROID consortium (UC Berkeley et al.) |
| primary_source | DROID paper + official release; convenient port [IPEC-COMMUNITY/droid_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/droid_lerobot) (~92,223 ep / 27,044,326 frames / 15 FPS). Newer Hub: [lerobot/droid_1.0.1](https://huggingface.co/datasets/lerobot/droid_1.0.1) (~95,658 ep) |
| bucket / legal | FOUNDATION / CONDITIONAL |
| purpose | manipulation |
| embodiment | FRANKA |
| scale | **VERIFIED** ~92k–96k episodes depending on port |
| video | exterior + wrist **ESTIMATE** (DROID standard); 15 FPS |
| proprio / actions | robot state + action |
| tactile_contact | none |
| language | 31k+ unique task strings (v2 port) |
| NCES / ECD | 4 / 4 |
| stage | 2–5 |
| license | Paper: **CC BY 4.0**. Several LeRobot ports tag Apache-2.0 (Hub default). **Do not treat the tag as upstream.** |
| commercial_safe | CONDITIONAL until mix locks the paper term |
| license_verified | 2026-08-28, paper vs Hub tag mismatch recorded |
| derived_from | original DROID TFDS (~2 TB) → LeRobot (~400 GB) **ESTIMATE** from community card |
| priority | P0 |
| unique_information | diverse real robot (Franka) manipulation |
| notes | Complements HiFi. LEGAL stays CONDITIONAL until commercial counsel accepts paper CC BY over mirror Apache. |

### 2.3 `oxe`

| field | value |
| --- | --- |
| source_org | Open X-Embodiment / RT-X (Google et al.) |
| primary_source | [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/) + original TFDS. Hub [jxu124/OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) is **unofficial** |
| bucket / legal | FOUNDATION / UNRESOLVED |
| purpose | manipulation / cross-embodiment |
| embodiment | MULTI (~22 bodies, 1M+ traj **VERIFIED** on unofficial card) |
| scale | 1M+ real trajectories **VERIFIED** (project claim) |
| video / proprio / actions | heterogeneous per constituent |
| NCES / ECD | 5 / 4 |
| stage | 4–5 |
| license | **No single license.** Unofficial Hub CC BY 4.0 is not 55 sources. Constituents include CC BY, NC, research-only, and undocumented. |
| commercial_safe | NO at family grain. Per-subset manifest required. Mixture key `oxe_cc_by` is a **wish**, not an audit. |
| license_verified | 2026-08-28, family UNRESOLVED |
| derived_from | union of 50+ robot datasets |
| priority | P0 |
| unique_information | cross-embodiment robot action → NCES stress test |
| notes | Empirical test of NCES. Commercial mix must list allowed OXE children, not “OXE”. |

### 2.4 `bones-seed`

| field | value |
| --- | --- |
| source_org | Bones Studio |
| primary_source | [bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed), license [bones.studio/info/seed-license](https://bones.studio/info/seed-license) |
| bucket / legal | FOUNDATION / RESEARCH_ONLY |
| purpose | motion (whole-body) |
| embodiment | HUMAN (SOMA) + G1 MuJoCo CSV in the same family |
| scale | **VERIFIED** 142,220 motions; ~288 h @ 120 fps; 522 actors |
| video | no RGB foundation layer (mocap) |
| proprio / actions | BVH + G1 joint CSV |
| language | up to 6 NL descriptions + temporal segments |
| NCES / ECD | 5 / 4 |
| stage | 2–4 |
| license | BONES-SEED Agreement: academic or qualifying startup (< $1M revenue); commercial license via licensing@bones.studio |
| commercial_safe | NO |
| license_verified | 2026-08-28, primary license page |
| derived_from | original Vicon capture |
| priority | P0 |
| unique_information | human whole-body motion prior |
| notes | Gated Hub. **All G1 retargets of this corpus inherit RESEARCH_ONLY** until Bones says otherwise. |

### 2.5 `agency-nullxes`

| field | value |
| --- | --- |
| source_org | NULLXES |
| primary_source | *does not exist yet* |
| bucket / legal | FOUNDATION / COMMERCIAL |
| purpose | agency |
| embodiment | MULTI (labels on converted episodes) |
| scale | **0** |
| video / proprio / actions | inherited from host episode |
| tactile_contact | optional |
| language | includes `instruction = NULL` |
| NCES / ECD | 2 / 2 (labels, not kinematics) |
| stage | 7–8 |
| license | NULLXES original annotations |
| commercial_safe | YES (our labels; host video still follows host LEGAL) |
| license_verified | n/a (unborn) |
| derived_from | host corpora + deterministic events + heuristics + classifiers + gold review |
| priority | P0 |
| unique_information | intervention / deliberate non-intervention (ACT / OBSERVE / HOLD / WAIT, DEFER/BLOCK/ALLOW, persistence, recovery) |
| notes | **Red hole.** Open robotics data gives obs/action/next/task. It does not give agency. Not one classifier. |

### 2.6 `simulation-nullxes`

| field | value |
| --- | --- |
| source_org | NULLXES |
| primary_source | *does not exist yet* |
| bucket / legal | FOUNDATION / COMMERCIAL |
| purpose | dynamics |
| embodiment | SIM (and real transitions as targets) |
| scale | **0** |
| actions | branched ACT A/B, HOLD, WAIT → distinct Z futures |
| NCES / ECD | 4 / 4 |
| stage | 6, 9 |
| license | NULLXES sim + any inbound mesh licenses |
| commercial_safe | YES if assets are |
| derived_from | physics engine + our scenarios; ArtVIP etc. are optional props |
| priority | P0 |
| unique_information | branched counterfactual futures |
| notes | **Red hole.** Imitation corpora cannot teach EAC. This is where `L_future` gets real branches. |

### 2.7 `humanoid-everyday`

| field | value |
| --- | --- |
| source_org | USC PSI Lab |
| primary_source | [USC-PSI-Lab/humanoid-everyday](https://huggingface.co/datasets/USC-PSI-Lab/humanoid-everyday), paper arXiv:2510.08807. G1 slice: `USC-PSI-Lab/Humanoid-Everyday-G1` |
| bucket / legal | FOUNDATION / COMMERCIAL |
| purpose | manipulation + loco-integrated |
| embodiment | G1 |
| scale | 260+ tasks, 7 categories **VERIFIED** (card). Episode-hours **ESTIMATE** — Hub size_category 1K–10K is not hours |
| video | yes |
| proprio / actions | teleop whole-ish G1 |
| language | task descriptions |
| NCES / ECD | 4 / 4 |
| stage | 4–5 |
| license | Apache-2.0 **VERIFIED** (Hub tag) |
| commercial_safe | YES **CONDITIONAL on confirming card text matches Apache for data, not only code** |
| derived_from | original teleop |
| priority | P1 |
| unique_information | open-world G1: dexterous + human–humanoid + locomotion-integrated |
| notes | Pair with BONES (human prior) rather than replacing it. |

### 2.8 `unifolm-wbt`

| field | value |
| --- | --- |
| source_org | Unitree Robotics |
| primary_source | UnifoLM-WBT collection under `unitreerobotics/` (public from 2026-03-05 per press). Example Hub names: `G1_WBT_Brainco_*`, `G1_WB_Dex5_*`. Third-party token dumps (`mertalbaba/UnifoLM-WBT-Hand-*`) are **not** primary; several marked DEPRECATED |
| bucket / legal | FOUNDATION / CONDITIONAL |
| purpose | motion + manipulation (whole-body) |
| embodiment | G1 |
| scale | growing task collection; frame counts 10^4–10^5 per task **ESTIMATE** (press tables) |
| video | yes (head/wrist typical) |
| proprio / actions | legs, waist, arms, hands (Brainco / Inspire / Dex variants) |
| NCES / ECD | 5 / 5 |
| stage | 2–5 |
| license | per-repo Apache-2.0 common on Unitree LeRobot cards **UNVERIFIED as collection-wide** |
| commercial_safe | CONDITIONAL until collection LICENSE is read per child |
| derived_from | original Unitree teleop (not BONES) |
| priority | P1 |
| unique_information | real G1 whole-body teleoperation |
| notes | This is robot whole-body **grounding**. BONES is the human prior. Do not collapse them. Skip deprecated Hand-v1/v2 token repos. |

### 2.9 `unitree-g1-dex-teleop`

| field | value |
| --- | --- |
| source_org | Unitree Robotics |
| primary_source | `unitreerobotics/G1_Dex3_*` and `G1_Dex1_*` (ToastedBread, BlockStacking, Pouring, GraspSquare, …) |
| bucket / legal | FOUNDATION / COMMERCIAL |
| purpose | manipulation |
| embodiment | G1 + Dex1/Dex3 |
| scale | many mid-size LeRobot sets (100K–1M rows band on Hub) **ESTIMATE** |
| video | AVP teleop recipe; scene must match first frame (card caveat) |
| NCES / ECD | 4 / 4 |
| stage | 3–5 |
| license | Apache-2.0 **VERIFIED** on sampled cards |
| commercial_safe | YES |
| derived_from | original Unitree |
| priority | P1 |
| unique_information | official G1 dexterous task suite (not community 50-ep clones) |
| notes | One family. Do not list each Dex3 task as its own registry row. |

### 2.10 `mosaic-g1`

| field | value |
| --- | --- |
| source_org | BAAI-Humanoid |
| primary_source | [BAAI-Humanoid/MOSAIC_Dataset](https://huggingface.co/datasets/BAAI-Humanoid/MOSAIC_Dataset), arXiv:2602.08594 |
| bucket / legal | FOUNDATION / CONDITIONAL |
| purpose | motion (retarget) |
| embodiment | HUMAN (AMASS-style) + G1 NPZ |
| scale | small on Hub size tag (`n<1K`) **UNVERIFIED hours** — still unique because **paired** |
| NCES / ECD | 5 / 4 |
| stage | 4 |
| license | CDLA Permissive 2.0 **VERIFIED** (Hub). Inbound AMASS constituents may be stricter. |
| commercial_safe | CONDITIONAL (AMASS chain) |
| derived_from | AMASS-style human + retarget to G1 |
| priority | P1 |
| unique_information | paired human → humanoid retargeting |
| notes | Converter-validation gold for human→NCES→G1. Scale may stay below Foundation hours — uniqueness still P1. |

### 2.11 `deco-50`

| field | value |
| --- | --- |
| source_org | BAAI-Humanoid |
| primary_source | [BAAI-Humanoid/DECO-50](https://huggingface.co/datasets/BAAI-Humanoid/DECO-50), arXiv:2602.05513 |
| bucket / legal | FOUNDATION / COMMERCIAL |
| purpose | contact |
| embodiment | dual-arm real |
| scale | **VERIFIED** 50 h, 4 scenes, 28 subtasks, >5M frames |
| tactile_contact | **yes** (plugin tactile) |
| NCES / ECD | 3 / 3 |
| stage | 3, 6 |
| license | Apache-2.0 **VERIFIED** (Hub) |
| commercial_safe | YES |
| derived_from | original teleop |
| priority | P1 |
| unique_information | tactile bimanual interaction |
| notes | Rarest open contact layer at useful hours. Do not drown it in OXE. |

### 2.12 `robomind`

| field | value |
| --- | --- |
| source_org | X-Humanoid / RoboMIND |
| primary_source | [x-humanoid-robomind/RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND), arXiv:2412.13877. V2 pointed at ModelScope |
| bucket / legal | FOUNDATION / CONDITIONAL |
| purpose | manipulation |
| embodiment | MULTI |
| scale | Hub `n>1T` **VERIFIED** as size category, not as token count. Gated `auto` |
| NCES / ECD | 4 / 4 |
| stage | 4–5 |
| license | Apache-2.0 **VERIFIED** (Hub tag). Gated access. |
| commercial_safe | CONDITIONAL (gate + confirm data vs code license) |
| derived_from | original multi-robot teleop |
| priority | P1 |
| unique_information | multi-embodiment real teleop at very large robot-data scale |
| notes | Overlaps OXE philosophically; keep if morphology coverage is broader or cleaner. |

### 2.13 `bridge-v2`

| field | value |
| --- | --- |
| source_org | RAIL / BridgeData |
| primary_source | original BridgeData V2; ports e.g. [jesbu1/bridge_v2_lerobot](https://huggingface.co/datasets/jesbu1/bridge_v2_lerobot) (~53k ep, WidowX, 5 FPS) |
| bucket / legal | FOUNDATION / CONDITIONAL |
| purpose | manipulation |
| embodiment | WIDOWX |
| scale | ~50k–60k episodes **VERIFIED** on ports |
| license | upstream often CC BY 4.0; Hub ports Apache-2.0 — same DROID-style tag drift |
| commercial_safe | CONDITIONAL |
| derived_from | original Bridge; also an OXE constituent |
| priority | P2 |
| unique_information | WidowX real kitchen-style; **duplicate of OXE unless OXE subset excludes it** |
| notes | Lower priority if OXE CC-BY child list already includes Bridge. |

### 2.14 `agibot-world`

| field | value |
| --- | --- |
| source_org | AgiBot |
| primary_source | [agibot-world/AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026), Alpha [agibot-world/AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) |
| bucket / legal | FOUNDATION / RESEARCH_ONLY |
| purpose | manipulation |
| embodiment | dual-arm real |
| scale | large (Alpha hundreds of hours historically; 2026 set is the current drop) **ESTIMATE** |
| license | CC BY-NC-SA 4.0 **VERIFIED** (Hub) |
| commercial_safe | NO |
| derived_from | original AgiBot |
| priority | P1 research mix only |
| unique_information | large dual-arm real household-scale scenes |
| notes | Already in `license.py` commercial block. Share-alike also infects derivatives. |

### 2.15 `ego4d`

| field | value |
| --- | --- |
| source_org | Ego4D Consortium |
| primary_source | [ego4d-data.org](https://ego4d-data.org/) — not random Hub mirrors |
| bucket / legal | FOUNDATION / RESEARCH_ONLY |
| purpose | observation |
| embodiment | HUMAN |
| scale | ~3,600 h **VERIFIED** (project) |
| video | egocentric, in-the-wild |
| proprio / actions | no robot NCES |
| NCES / ECD | 1 / 1 |
| stage | 1 |
| license | Ego4D custom (non-transferable; research/product-dev clauses). Redistribution of video is restricted. |
| commercial_safe | NO for AMATERASU commercial mix (weights trained on Ego4D video need counsel) |
| derived_from | original |
| priority | P1 |
| unique_information | in-the-wild egocentric physical observation |
| notes | Commercial mix already `drop: ego4d`. Stage-1 research only. |

### 2.16 `epic-kitchens`

| field | value |
| --- | --- |
| source_org | University of Bristol / EPIC |
| primary_source | EPIC-KITCHENS-100 |
| bucket / legal | FOUNDATION / RESEARCH_ONLY |
| purpose | observation |
| embodiment | HUMAN |
| scale | 100 h **VERIFIED** |
| license | CC BY-NC 4.0 **VERIFIED** on HF M4 card |
| commercial_safe | NO |
| NCES / ECD | 1 / 1 |
| stage | 1 |
| priority | P2 |
| unique_information | dense kitchen egocentric (weaker than Ego4D scale, stronger task structure) |
| notes | Commercial `drop: epic`. |

### 2.17 `robocoin-g1`

| field | value |
| --- | --- |
| source_org | RoboCOIN |
| primary_source | `RoboCOIN/Unitree_G1edu_u3_*` (place_bread ~1464 ep, pour_drink ~1298 ep, …) |
| bucket / legal | FOUNDATION / COMMERCIAL |
| purpose | manipulation |
| embodiment | G1edu-u3 |
| scale | hours-class if summed **ESTIMATE**; single tasks are already larger than community toys |
| license | Apache-2.0 **VERIFIED** on sampled cards |
| commercial_safe | YES |
| NCES / ECD | 4 / 4 |
| stage | 3–5 |
| priority | P2 |
| unique_information | G1edu real tasks at non-toy episode counts |
| notes | One family. Sum children; do not 40-row the Hub. |

### 2.18 `sonic-g1-tracking`

| field | value |
| --- | --- |
| source_org | community / NVlabs SONIC lineage |
| primary_source | [gserifi/sonic-tracking-bones-seed](https://huggingface.co/datasets/gserifi/sonic-tracking-bones-seed) (~129,753 ep, LeRobot v3). Related: `GeorgiaTech/g1_bones_seed_sonic_129k_50hz` |
| bucket / legal | QA / RESEARCH_ONLY |
| purpose | dynamics |
| embodiment | G1 29-DoF |
| scale | ~129k clips, ~47.6M frames, ~112 GB **VERIFIED** (card) |
| license | **inherits BONES-SEED**. GeorgiaTech derived set states private + upstream terms. |
| commercial_safe | NO |
| derived_from | `bones-seed` → retarget → SONIC rollout |
| NCES / ECD | 4 / 4 |
| stage | 6, 9 |
| priority | P2 |
| unique_information | closed-loop G1 tracking of human motions (policy physics, not new human prior) |
| notes | Provenance graph demo: Apache-looking LeRobot wrap does not free BONES. |

### 2.19 `amass-smpl`

| field | value |
| --- | --- |
| source_org | AMASS / MPI |
| primary_source | AMASS website; Hub derivatives e.g. `yan0116/SMPL_Humanoid_offline_dataset` |
| bucket / legal | QA / CONDITIONAL |
| purpose | motion |
| embodiment | HUMAN (SMPL) |
| unique_information | generic SMPL mocap; **mostly duplicate of BONES if BONES is used** |
| NCES / ECD | 4 / 2 |
| priority | P3 |
| notes | Admit only if BONES is unavailable or AMASS covers a motion class BONES lacks. |

### 2.20 `artvip`

| field | value |
| --- | --- |
| source_org | X-Humanoid |
| primary_source | [X-Humanoid/ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP) |
| bucket / legal | QA / COMMERCIAL |
| purpose | dynamics (assets) |
| embodiment | SIM |
| scale | 476 articulated objects, 6 scenes **VERIFIED** |
| license | Apache-2.0 **VERIFIED** |
| NCES / ECD | 2 / 3 |
| stage | 6, 9 |
| priority | P2 |
| unique_information | physics-faithful articulated object twins for our sim, not a trajectory corpus |
| notes | Prop library for `simulation-nullxes`, not a mixture weight. |

### 2.21 `libero`

| field | value |
| --- | --- |
| source_org | LIBERO |
| primary_source | original LIBERO release |
| bucket / legal | QA / CONDITIONAL |
| purpose | manipulation |
| embodiment | FRANKA tabletop |
| unique_information | language-conditioned tabletop; **OXE/DROID-adjacent duplicate** |
| license | datasets often CC BY 4.0; code MIT |
| priority | P3 |
| notes | Converter QA / eval, not foundation hours. |

### 2.22 `openeai-aggregate`

| field | value |
| --- | --- |
| source_org | OpenEAI |
| primary_source | [OpenEAI/OpenEAI-Dataset](https://huggingface.co/datasets/OpenEAI/OpenEAI-Dataset) (~3.12 TB, MIT tag) |
| bucket / legal | QA / UNRESOLVED |
| purpose | **schema reference** |
| unique_information | someone else’s OXE/UMI unification — steal the schema, not the pixels |
| commercial_safe | NO until every inbound child is mapped |
| priority | P2 |
| notes | **Do not train AMATERASU on this blob.** Use as Data Plane design reference. |

### 2.23 `contact-tactile-open`

| field | value |
| --- | --- |
| source_org | mixed (ContactPose, FeelAnyForce, Digit, …) |
| primary_source | per child — **not audited this pass** |
| bucket / legal | QA / UNRESOLVED |
| purpose | contact |
| unique_information | residual tactile after DECO-50 |
| NCES / ECD | 2 / 2 |
| priority | P2 |
| notes | Placeholder family. Next audit fills children or kills the row. |

### 2.24 `ego-hand-micro`

| field | value |
| --- | --- |
| source_org | mixed small LeRobot ego-hand sets (RoboX-class, ~28 demos) |
| bucket / legal | QA / UNRESOLVED |
| purpose | manipulation |
| scale | tens of demos |
| unique_information | converter gold: human hand demo → NCES |
| priority | P2 |
| notes | Never a 32B mixture component. |

### 2.25 `ikea-g1-challenge`

| field | value |
| --- | --- |
| source_org | BitRobot |
| primary_source | [BitRobot/2026-humanoid-ikea-assembly-challenge](https://huggingface.co/datasets/BitRobot/2026-humanoid-ikea-assembly-challenge) |
| bucket / legal | QA / COMMERCIAL |
| license | CC BY 4.0 **VERIFIED** (Hub) |
| unique_information | long-horizon G1 assembly (tiny N, ROS/MCAP) |
| priority | P3 |
| notes | Horizon/memory eval, not pretrain. |

### 2.26 `community-g1-lerobot`

| field | value |
| --- | --- |
| source_org | Hub users |
| primary_source | `*unitree_g1*pick*` / 50–100 episode clones |
| bucket / legal | REJECT / UNRESOLVED |
| unique_information | **none** vs Unitree official + RoboCOIN + Humanoid Everyday |
| notes | Default reject. Promotion to QA only with hours + ECD + license. |

### 2.27 `habitat-humanoids`

| field | value |
| --- | --- |
| source_org | Meta AI Habitat |
| primary_source | [ai-habitat/habitat_humanoids](https://huggingface.co/datasets/ai-habitat/habitat_humanoids) |
| bucket / legal | REJECT / RESEARCH_ONLY |
| license | CC BY-NC-SA 4.0 **VERIFIED** |
| unique_information | none for NCES action (avatars/walk-reach) |
| notes | Not an action corpus. |

### 2.28 `g1-hardware-cad`

| field | value |
| --- | --- |
| source_org | LeRobot |
| primary_source | [lerobot/unitree-g1-hardware-modifications](https://huggingface.co/datasets/lerobot/unitree-g1-hardware-modifications) (seen 2026-08-28) |
| bucket / legal | REJECT / COMMERCIAL |
| unique_information | none (geometry, not motion) |
| notes | Watchlist false positive. |

### 2.29 `egodex`

| field | value |
| --- | --- |
| source_org | EgoDex |
| primary_source | original EgoDex release |
| bucket / legal | QA / RESEARCH_ONLY |
| purpose | manipulation |
| embodiment | HUMAN |
| commercial_safe | NO (`license.py` commercial drop) |
| unique_information | ego dexterous; likely dominated by HiFi at foundation scale |
| priority | P3 |
| notes | Keep for overlap check vs HiFi; do not dual-count. |

---

## 3. Coverage matrix (v0.1)

Axes the goddess actually needs. Cell = strongest family, not a wish.

Legend: **S** strong · **W** weak · **G** gap (including scale=0)

|  | observe | act | NULL instruction | contact | long horizon |
| --- | :---: | :---: | :---: | :---: | :---: |
| **human, real, short** | S Ego4D/EPIC (research) · W HiFi | S HiFi | G | W HiFi visual contact | G |
| **human, whole-body** | W (mocap, little RGB) | S BONES | G | W | W |
| **Franka, real** | S DROID | S DROID | G | G | W |
| **multi-robot, real** | S OXE/RoboMIND | S OXE | G | W | W |
| **G1, real, arms** | S Unitree Dex / RoboCOIN / Everyday | S same | G | W | W IKEA |
| **G1, real, whole-body** | S UnifoLM-WBT | S UnifoLM-WBT | G | W | W |
| **human→G1 paired** | W MOSAIC | S MOSAIC | G | G | G |
| **tactile** | S DECO-50 | S DECO-50 | G | S DECO-50 | G |
| **sim, branched futures** | G | G | G | W ArtVIP assets | G |
| **agency (ACT/HOLD/WAIT)** | G | G | G | G | G |

Collapsed into EAFM language:

| Capability | Coverage |
| --- | --- |
| How a human hand manipulates | **S** HiFi |
| How Franka manipulates | **S** DROID |
| Many robots → one NCES | **S** OXE (legal messy) |
| Human whole-body prior | **S** BONES (not commercial) |
| G1 whole-body grounding | **S** UnifoLM-WBT + Everyday |
| Tactile | **W/S** DECO-50 only |
| In-the-wild observe | **S** Ego4D — **G** commercial-safe substitute |
| `instruction = NULL` | **G** |
| Deliberate non-intervention | **G** (`agency-nullxes`) |
| Counterfactual Z futures | **G** (`simulation-nullxes`) |
| Long episode + memory + recovery | **G** |

Commercial mix 32/28/18/12/10 is a **manipulation profile**. The matrix says why it cannot be the eternal diet: 78% act-imitation, almost no NULL, no branches, no agency.

---

## 4. Data gaps (ordered)

1. **Agency labels on real episodes** — OBSERVE/HOLD/WAIT/ACT, persistence, recovery, DEFER/BLOCK/ALLOW. Host data exists; labels do not.
2. **Branched sim futures** — same Z, four interventions, four futures.
3. **Commercial-safe egocentric observation** — without Ego4D/EPIC, stage-1 commercial is HiFi-only vision.
4. **NULL-instruction long episodes** — idle, wait-for-human, resume.
5. **Tactile beyond 50 h / 4 scenes.**
6. **OXE child license manifest** — blocks commercial `oxe_cc_by` from being honest.
7. **DROID / Bridge Hub-tag vs paper-term reconcile.**

Gaps 1–2 are datasets we **generate**. Gaps 3–6 are audits + filters. Gap 7 is legal hygiene.

---

## 5. Provenance rules (non-negotiable)

```text
LEGAL(child) = meet( LEGAL(parent_1), LEGAL(parent_2), ... )
```

Worked examples:

- `sonic-g1-tracking` ← BONES → `RESEARCH_ONLY` even if LeRobot says Apache-2.0.
- MOSAIC ← AMASS constituents → `CONDITIONAL`.
- OpenEAI ← OXE + UMI children → `UNRESOLVED` until mapped.
- Agency labels on HiFi video → labels COMMERCIAL, pixels follow HiFi (COMMERCIAL). Agency labels on Ego4D → pixels RESEARCH_ONLY.

`configs/data/mixture_commercial.json` is a **profile**, not a license oracle. The oracle is this registry + a future per-file license manifest.

---

## 6. What this registry does *not* do

- Download or convert anything.
- Change the 31,740,290,560 freeze.
- Promote OpenEAI, community G1 pick-place, or CAD to FOUNDATION.
- Invent hours where Hub only has size categories.

---

## 7. Next sequence (locked)

```text
Registry v0.1          ← you are here
        ↓
Coverage matrix freeze (this §3; iterate only with new families)
        ↓
AMATERASU Agency Dataset Spec v0.1
  (label taxonomy, host corpora, gold subset size, NULL policy)
        ↓
AMATERASU Dynamics/Sim Dataset Spec v0.1
  (branching protocol, NFE, engine, asset licenses)
        ↓
OXE subset license manifest (commercial-safe children only)
        ↓
Converter contracts per P0/P1 FOUNDATION family
        ↓
Converters (HiFi, DROID, BONES/G1, …)
```

Do not write converters until Agency/Dynamics specs exist. Otherwise the mixer will happily emit imitation-only samples and call it EAFM.

Watchlist orgs (weekly, family grain): `simple-world-lab`, `bones-studio`, `unitreerobotics`, `USC-PSI-Lab`, `BAAI-Humanoid`, `X-Humanoid`, `x-humanoid-robomind`, `RoboCOIN`, `agibot-world`, `lerobot`, `NVlabs`.
