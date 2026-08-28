# AMATERASU Agency Dataset Specification v0.1

**Author:** MagistrTheOne | NULLXES  
**Date:** 28 August 2026  
**Status:** FROZEN as the data contract for Stages 7–8 (and ACPC labels for Stage 6/9)  
**Does not:** download data, write converters, change the 31,740,290,560 freeze, or add a teacher-LLM as the source of agency

Companion docs: architecture [AMATERASU_EAFM_ARCHITECTURE_v0.1.md](AMATERASU_EAFM_ARCHITECTURE_v0.1.md) §H–J, registry [AMATERASU_DATASET_REGISTRY_v0.1.md](AMATERASU_DATASET_REGISTRY_v0.1.md). Hub family shopping is **stopped** at 48. This spec owns what those families **lack**.

Claim tags: **VERIFIED** (matches frozen enums/losses), **ESTIMATE**, **HYPOTHESIS**.

---

## 0. Decision

There are **two** NULLXES corpora, not one blob:

| Corpus | Full name | Unique information |
| --- | --- | --- |
| **AAC** | AMATERASU Agency Corpus | intervention vs deliberate non-intervention on **factual** timelines |
| **ACPC** | AMATERASU Counterfactual Physical Corpus | **sibling futures** from one `Z_t` |

Registry rows `agency-nullxes` and `simulation-nullxes` (`scale = 0`) are these two. They start **now**, in parallel with converter contracts — not after every Hub download.

Operational shortlist (train vs reserve) is §1. Everything else in the 48 is QA / evaluation / legal-pending.

---

## 1. Operational shortlist (not 48 downloads)

### 1.1 DOWNLOAD / TRAIN (commercial path, after named legal gates)

| Role | Family | Gate before bytes |
| --- | --- | --- |
| Human manipulation | `hifi-umi-2k` | CC BY 4.0 already **VERIFIED** |
| Robot manipulation | `droid` | paper CC BY vs Hub Apache **reconcile** |
| Cross-embodiment | `oxe` **children only** | per-subset license manifest |
| Real G1 body | `unifolm-wbt` | per-child Apache vs collection |
| G1 dexterity | `unitree-g1-dex-teleop` | Apache **VERIFIED** sampled |
| Robot tactile | `deco-50` | Apache **VERIFIED** |
| Commercial ego | `egosuite-open100k` | full `commercial-training-no-resale` text |
| Human tactile seed | `ego-tactile-opengraph` | CC BY; **weight ≈ 0** in mixture (seed) |
| Counterfactual factory | ManiSkill / RoboCasa / CALVIN / RLBench **as environments** | not Hub MimicGen blobs |

### 1.2 RESEARCH TRAIN (never commercial mix)

`bones-seed`, `ego4d`, `agibot-world`, `interndata-a1`, `mpi-hoi`, `realsource-world` (NC-SA), `humanoid-everyday` if remaining Apache audit fails.

### 1.3 QA / REFERENCE (do not “must download”)

GRAIL, HRDexDB, FEEL, Firstly, OpenTouch (license unresolved), OpenEAI schema, IKEA, community G1, Nemotron math except Proofs-v2 **template**.

OpenGraph stays **FOUNDATION seed**: unique information is fundamental; 1.28 h does not get a sampling slice.

### 1.4 P0 vs P1 for Data Plane start

**P0 now:** HiFi, DROID (after reconcile), audited OXE children, BONES (research), UnifoLM/Unitree Dex, DECO, EgoSuite (after license text), OpenGraph seed, resettable sim **engines**, **AAC + ACPC design** (this document).

**P1 after converter contract exists:** Humanoid Everyday, MOSAIC, MPI-HOI, NVIDIA G1 Locomanip, RoboMIND.

---

## 2. Frozen ontology

Label unit = **slow tick** (architecture: ~2–5 Hz). Fast ticks inherit the last slow decision until interrupt (contact spike, tracking loss, speech).

Chain **VERIFIED** against `IntentKind` / `GateDecision`:

```text
Z_t, A_t, M_t, G_t (nullable), E_t
        │
        ▼
CANDIDATES  I_1…I_K  + OBSERVE + HOLD + WAIT     (K = 8 ACT slots)
        │
        ▼
INTENT*     selected kind                         IntentKind 0–10
        │
        ▼
GATE        ALLOW | DEFER | BLOCK                 GateDecision 0–2
        │
        ▼
INTERVENTION  executed family:
              ACT (flow on) | HOLD (flow on) | OBSERVE (flow off) | WAIT (flow off)
        │
        ▼
CONSEQUENCE  factual Z_{t+k}  and/or  sibling Ẑ
        │
        ▼
VERIFIER    physical bits + calibration target    (PVR; not LLM judge)
```

NOOP is the EAC umbrella over OBSERVE / HOLD / WAIT. HOLD is **not** zero torque.

`G_t` may be NULL. `null_instruction=true` is a first-class sample flag (already on `AMATERASUSample`).

### 2.1 Intent kinds (**VERIFIED** `amaterasu.tensors.modality.IntentKind`)

| id | name | meaning |
| ---: | --- | --- |
| 0–7 | `ACT_0…ACT_7` | candidate physical interventions (task-space NCES targets, not joint recipes) |
| 8 | `OBSERVE` | do not intentionally modify the world; keep estimating |
| 9 | `HOLD` | maintain current NCES; low-level balance/grasp stays on |
| 10 | `WAIT` | defer a **selected** future ACT until time or predicted state change |

Teacher may leave ACT slot identity unspecified (`ACT_0` as “some ACT”) when segmentation cannot distinguish I_k. Gold subset should prefer named slots when two ACT candidates are both plausible.

### 2.2 Gate (**VERIFIED** `GateDecision`)

| id | name | meaning |
| ---: | --- | --- |
| 0 | `ALLOW` | constraint layer permits the selected intent |
| 1 | `DEFER` | not now (premature, wait-for-event, missing info) |
| 2 | `BLOCK` | unsafe / irreversible / ECD-incapable / human-priority |

If every ACT is BLOCK or DEFER → executed family is OBSERVE or HOLD (architecture §H).

### 2.3 Justification class (AAC-only, for `L_nonint` / mining)

Not an extra network head in the freeze. Stored as `justification` on the sample. **HYPOTHESIS** mapping:

| justification | typical INTENT* | typical GATE on ACT candidates |
| --- | --- | --- |
| `justified` | ACT_k | ALLOW |
| `unnecessary` | OBSERVE or WAIT | BLOCK or DEFER |
| `premature` | WAIT | DEFER |
| `unsafe` | OBSERVE or HOLD | BLOCK |
| `insufficient_info` | OBSERVE | DEFER |
| `continue_observation` | OBSERVE | DEFER |

These six must **all** appear in AAC. A corpus of only `justified` is imitation with extra tokens.

### 2.4 Persistence / interruption (AAC event tags)

Boolean / enum on the window, not new IntentKind values:

`persist` | `abandon` | `human_interrupt` | `env_change` | `recovery` | `uncertainty_high`

Stage 8 consumes long traces where these flip.

---

## 3. AAC — AMATERASU Agency Corpus

**Factual** timelines only. One realized future per tick. Counterfactual siblings live in ACPC.

### 3.1 Hosts (relabel, do not invent physics)

| Host | What we steal | What we add |
| --- | --- | --- |
| HiFi, DROID, OXE children, G1 Dex/WBT, Everyday | cameras, NCES-able state, language | idle gaps, NULL, justification |
| EgoSuite | long ego observe | windows with **no** robot (pure OBSERVE prior) |
| DECO / OpenGraph | contact onsets | HOLD vs ACT at grasp |
| RealSource (research) | long dual-arm | recovery / quality-fail → abandon |
| NULLXES capture / sim logs | boring rooms | designed IOR |

### 3.2 Label pipeline (order is mandatory)

```text
RAW EPISODE
  │
  ├─ 1. deterministic events     contact onset/offset, gripper Δ, human bbox enter/leave,
  │                              speech onset, goal token appear/disappear, timeout
  │
  ├─ 2. physical heuristics      no-motion + no-contact → OBSERVE;
  │                              stable grasp + zero EE cmd → HOLD;
  │                              selected ACT with countdown/predicate → WAIT
  │
  ├─ 3. optional classifiers     vision/event models as *proposals*, never as ground truth
  │
  ├─ 4. optional teacher         only to propose justification; never to invent WAIT
  │                              because “the LLM would wait”
  │
  └─ 5. gold calibration         human review on a locked subset
```

Forbidden: “ask a VLM what the robot should have done” as the primary label. That recreates instruction-following.

### 3.3 Boring episodes (required class)

Robotics datasets treat these as trash. AAC treats them as **gold**.

Example (60 s, zero task completion):

```text
00:00 human enters
00:03 OBSERVE
00:08 nothing relevant
00:17 human moves an object → Z update, still OBSERVE
00:35 no reason to intervene
00:51 human leaves
01:00 still OBSERVE / WAIT
```

Target distribution inside the episode: OBSERVE/WAIT, not a single ACT. If the mix is `camera on → always ACT`, WAIT tokens will not survive training.

**ESTIMATE** share of AAC windows that are *boring* (no justified ACT in ±2 s): **≥ 25%**. Below that, `L_nonint` is decoration.

### 3.4 Intervention opportunity rate (IOR)

IOR = fraction of slow-tick windows where `justification == justified`.

**HYPOTHESIS freeze (to lock after Stage 7 curves):**

| Mix slice | IOR target | Non-intervention windows |
| --- | ---: | ---: |
| AAC overall | **0.30–0.45** | 0.55–0.70 |
| AAC boring-only subset | **≤ 0.05** | ≥ 0.95 |
| Manipulation-heavy hosts before relabel | often > 0.80 | **must be downsampled** |

Do **not** ship AAC with IOR ≈ 0.95. Mixer must reject a shard that exceeds IOR 0.60 after relabel **ESTIMATE** (hard fail in the future loader, not in this file as code).

Per-window justification histogram must contain all six classes; none may be < **2%** of AAC **ESTIMATE** (unsafe may be rare; if < 2%, oversample BLOCK sims).

### 3.5 Instruction mix

| flag | meaning | **ESTIMATE** AAC share |
| --- | --- | ---: |
| `null_instruction=true` | `G_t` empty | **≥ 40%** |
| explicit language | task string present | remainder |

Stage 7–8 without NULL trains an instruction slave.

---

## 4. ACPC — AMATERASU Counterfactual Physical Corpus

More important than AAC for `L_future` and PRLVR. Logged imitation **cannot** supply unused branches (architecture §J).

### 4.1 Unit

Resettable simulator (or replayable physics) at state `Z_t`:

```text
                 SAME Z_t
                    │
      ┌─────────────┼──────────────┬─────────────┐
      ▼             ▼              ▼             ▼
    ACT A         ACT B          WAIT          HOLD
      │             │              │             │
      ▼             ▼              ▼             ▼
  Z_A(t+k)      Z_B(t+k)      Z_W(t+k)      Z_H(t+k)
      │             │              │             │
   verifier      verifier       verifier      verifier
```

k ∈ {1, 2, 4, 8} (matches `L_future`). Store **all** siblings, including failures.

WAIT/HOLD branches are not “no op in the renderer.” HOLD keeps contact/balance; WAIT is freeze of **intent**, not necessarily freeze of the world (human may still move — then Z_W ≠ Z_t).

### 4.2 Engines (factory, not datasets)

ManiSkill, RoboCasa, CALVIN, RLBench, Isaac Lab (G1 locomanip scene family). ArtVIP / HiPHI meshes as **props**. InternData-A1 trajectories may initialize `Z_t` in **research** ACPC only (NC-SA).

**Forbidden:** train ACPC by reading `myconnects/robocasa-pretrain` successful demos as if they were siblings.

### 4.3 Verifier (PVR)

No LLM judge. Bits **HYPOTHESIS** (extend in Dynamics spec, do not drop):

`success` `stable_contact` `slip` `collision` `human_safety` `pose_err` `dropped` `timeout` `reversible`

Gate labels on siblings: FAIL+collision → BLOCK; WAIT sibling that avoids collision → DEFER/ALLOW OBSERVE.

### 4.4 Ownership

This spec **freezes the ACPC sample contract**. Engine lists, USD licenses, NFE, and domain randomization → **Dynamics/Sim Dataset Spec v0.1** next. Do not delay AAC/ACPC schema waiting for that file.

---

## 5. Sample contract (additive)

Existing `AMATERASUSample` keeps sensors/NCES. Agency/ACPC fields below are the v0.1 addendum. Loaders may omit them before Stage 7 (`agency_on=false`).

| field | AAC | ACPC | loss |
| --- | --- | --- | --- |
| `intent_label` | IntentKind | teacher on executed branch | `L_intent` |
| `gate_label` | GateDecision | per sibling | `L_gate` |
| `justification` | enum §2.3 | optional on executed | `L_nonint` |
| `null_instruction` | bool | bool | `L_lang` empty |
| `candidate_mask` | which of 8+3 were proposed | all four families present | — |
| `persist` / `abandon` / `human_interrupt` / `env_change` / `recovery` | flags | rare | Stage 8 |
| `z_future` | factual k-stack | **K siblings × k** | `L_future` |
| `sibling_id` | 0 | 0=A,1=B,2=WAIT,3=HOLD | PRLVR |
| `verifier` | optional on real | required | PRLVR / calibration |
| `ior_shard` | bookkeeping | — | mixer |
| `gold` | bool | bool | eval |

Textual rationale is **optional train-only** language; runtime EAC does not emit CoT (registry §8).

---

## 6. Mix and stages

Architecture agency bucket **ESTIMATE** 8–12% of examples after packing.

| Stage | AAC | ACPC | agency losses |
| ---: | --- | --- | --- |
| 1–5 | off | off | off |
| 6 | off | **on** (futures only) | `L_future` |
| 7 | **on** | on | `L_intent` `L_nonint` `L_gate` |
| 8 | long AAC + boring | long ACPC | + persistence/recovery |
| 9 | target-robot AAC | target-domain ACPC | sim2real under gate |

AAC commercial vs research follows **host** LEGAL (provenance graph). NULLXES labels are COMMERCIAL; Ego4D pixels are not.

Gold subset **ESTIMATE:** ≥ 5 000 slow-tick windows, stratified by justification × NULL × embodiment. Used for ECE / non-intervention P/R (Stage 7 bar).

---

## 7. Anti-patterns (hard)

1. Teacher-LLM as the agency definition.  
2. IOR > 0.60 after relabel.  
3. No boring class.  
4. Observational DROID/HiFi treated as counterfactual.  
5. Scalar reward as EAC (architecture forbids).  
6. WAIT implemented as “dropped action token.”  
7. Sampling OpenGraph like HiFi.  
8. More Hub families before this spec is implemented in the mixer.

---

## 8. Next

1. **Dynamics/Sim Dataset Spec v0.1** — ACPC engines, assets, k-horizon, domain rand, commercial asset licenses.  
2. Mixer IOR hard-fail + `justification` on collate.  
3. Converter contracts for P0 DOWNLOAD/TRAIN hosts (sensors only; agency fields come from AAC relabel).  
4. OXE child license manifest (blocks honest `oxe_cc_by`).

Do not write converters that emit `intent_label` from “the human kept moving so ACT.”
