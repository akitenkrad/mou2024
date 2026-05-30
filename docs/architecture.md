**English** | [日本語](architecture.ja.md)

# Architecture

## Repository structure

```
mou2024/
├── Cargo.toml                  # Rust workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (mou2024-workspace)
├── simulation/                 # Rust crate hisim-simulation (bin hisim)
│   ├── src/
│   │   ├── main.rs             # CLI (clap: run / sweep / reproduce)
│   │   ├── config.rs           # Config, AbmModel, NetworkKind, LlmSettings
│   │   ├── world.rs            # HiSimWorld (WorldState), Tier, CoreState
│   │   ├── abm.rs              # BC / HK / SJ / Lorenz f_update + f_message
│   │   ├── prompts.rs          # core-tier action prompt
│   │   ├── parse.rs            # ACTION / MESSAGE parsing
│   │   ├── mechanisms.rs       # the 4 mechanisms (Environment / Decision / Mobilization / Aggregate)
│   │   ├── llm.rs              # Ollama→OpenAI fallback + cache + stance classifier + StanceMode (deterministic / external-LLM)
│   │   ├── bench.rs            # SoMoSiMu-Bench movement metrics + calibrated synthetic reference + alignment
│   │   ├── reproduce_mock.rs   # offline scripted client for `reproduce --mock`
│   │   ├── simulation.rs       # init_world + run driver + output writers
│   │   └── metrics.rs          # macro_bias / diversity / mobilized / polarization / core_influence
│   ├── examples/mock_smoke.rs  # offline (no live LLM) hybrid smoke
│   └── tests/integration_test.rs
├── tools/src/hisim_tools/      # Python package hisim-tools
│   ├── cli.py                  # subcommand dispatcher
│   ├── visualize.py            # attitude / mobilization / diversity-polarization / core-influence
│   ├── visualize_sweep.py      # ratio / model / network dependence
│   ├── show_experiment_settings.py
│   └── reproduce_paper.py      # Table 2/3 + SoMoSiMu-Bench report and figures
└── results/                    # runtime output (gitignored)
```

## `HiSimWorld` — the two-tier state

```rust
pub struct HiSimWorld {
    pub clock: SimClock,
    pub attitude: BTreeMap<AgentId, f64>,   // a ∈ [-1, 1], all agents (sorted keys)
    pub tier: BTreeMap<AgentId, Tier>,      // Core | Ordinary
    pub core: BTreeMap<AgentId, CoreState>, // profile + memory (core tier only)
    pub network: SocialNetwork,             // BA (default) / WS / ER
    pub current_event: String,              // exogenous event for the step
    pub macro_bias: f64,
    pub macro_diversity: f64,
    pub mobilized: usize,
}
```

`agent_ids()` returns the sorted `BTreeMap` keys, guaranteeing deterministic iteration in the socsim core. Tier assignment makes the top-`ceil(core_ratio · N)` highest-degree nodes the core tier (in scale-free networks, high degree = high influence).

## The four mechanisms (one per phase)

| Mechanism | Phase | Role |
|-----------|-------|------|
| `EnvironmentMechanism`   | Environment | Presents the trigger news / exogenous event of the step into `current_event` (deterministic). |
| `DecisionMechanism`      | Decision    | **Core tier** calls the LLM (the single LLM call site), parses the action, runs stance classification on broadcast messages, updates its own attitude and records the broadcast attitude. **Ordinary tier** does nothing here (its `f_selection` is the network neighbourhood gathered in the next phase). |
| `MobilizationMechanism`  | Interaction | Ordinary tier updates synchronously: snapshot start-of-step attitudes, build each agent's message set `M_i` from neighbours' `f_message(a_j)=a_j` plus core broadcasts from followed cores (one-way coupling), compute `Δa = f_update(a_i, M_i)`, write all back at once. |
| `AggregateMechanism`     | Reward      | Computes `macro_bias / diversity / mobilized / polarization`, records them, and `request_stop`s on convergence (T is enforced by the engine clock). |

The LLM call is confined to `DecisionMechanism`. With `--core-ratio 0.0` the core tier is empty, so no LLM call ever happens — the whole run is bit-deterministic.

## The ABM layer (`abm.rs`)

Following the unified `f_update / f_selection / f_message` formulation (Chuang & Rogers 2023, §6 of the design):

- `f_message(a_j) = a_j` — messages carry the sender's attitude unbiased.
- `f_update(a_i, M_i)` returns the **delta** `Δa`; the caller clamps `a_i + Δa` to `[-1, 1]` and writes back synchronously.

| Model | Rule | Qualitative behaviour |
|-------|------|-----------------------|
| **BC** (Deffuant) | `Δa = α·(m_j − a_i)` averaged over sources within the confidence bound ε | consensus |
| **HK** | move by α toward the mean of in-bound sources (incl. self) | consensus / clusters |
| **SJ** (Social Judgement) | assimilate in the acceptance region, repel in the rejection region | polarization |
| **Lorenz** | assimilate + reinforce + polarize (amplify extremes) | polarization |

The Ordinary-tier math mirrors the hegselmann2005 sibling's synchronous bounded-confidence update.

## Two RNG streams

```rust
const RNG_WORLD_INIT: u64 = 0; // network gen, attitude init, tier assignment
const RNG_ENGINE: u64    = 1;  // RandomActivationScheduler + ordinary ABM stochastics
```

Both derived from the root seed via `derive_seed`. The scheduler is `RandomActivationScheduler` (the order does not affect results because ordinary updates are synchronous).

## Stance annotation (`llm.rs`, `StanceMode`)

A core post's text is mapped to an attitude in `[-1, 1]` by one of two paths, selected with `--stance-annotator`:

- **`deterministic`** (default) — a lightweight keyword classifier (`classify_stance`, a stand-in for the paper's stance/sentiment annotation). Same text → same score, with **no extra LLM call**. This path is bit-identical to the prior behaviour, so the deterministic core stays reproducible and cache-consistent.
- **`llm`** — an external-LLM annotation path. The core LLM is asked to classify the post's stance on a 5-point scale (`stance_annotation_prompt`); the integer reply is mapped to `[-1, 1]` (`classify_stance_llm`), falling back to the deterministic classifier if no integer can be parsed. These annotation calls are counted in the metadata/budget and are cached, so `temperature=0` + fixed seed + the prompt cache keep them pseudo-deterministic on rerun.

## SoMoSiMu-Bench alignment (`bench.rs`)

`bench.rs` maps HiSim output to SoMoSiMu-Bench-style **movement metrics** (`MovementMetrics`: mobilization peak / peak step / final mobilization / final bias / final polarization / sustain ratio) and compares them to a per-movement reference, producing a structured `BenchComparison` with per-metric tolerance bands.

**Real vs calibrated (honest split):** the *observed* metrics are real simulator output. The *reference* is a **calibrated synthetic** curve (`reference_curve`, tagged `source = "calibrated-synthetic"`), parameterised to match the paper's qualitative descriptions of each movement (e.g. RoeOverturned shows stronger polarization than #MeToo) — it is **not** ground truth. The raw SoMoSiMu-Bench dataset is not bundled (acquisition/licensing sits with the original authors); when it is available, swap `BenchReference` for a CSV loader and the same alignment path applies unchanged. Comparisons check qualitative agreement (sign / order / band), not exact numerical match — the same philosophy as the `reproduce` anchors.

## `reproduce` — Table 2/3 in one shot

`hisim reproduce` runs, for the headline dataset, every ordinary-tier model in both the **hybrid** (LLM core + ABM periphery) and **pure-ABM** (`core-ratio 0`) regimes, aggregating final Bias / Diversity / Polarization / Mobilization and the mobilization gain (Table 3). It then aligns the pure-ABM movement dynamics for each dataset to the calibrated bench reference (Table 2). The result is `reproduce_summary.json` with observed-vs-paper anchors (PASS/off bands — e.g. *pure-ABM makes zero LLM calls*, *Lorenz/SJ polarize above BC*, *BC/HK stay in consensus*, *the LLM core amplifies mobilization*) plus the bench comparison, surfaced as a report and three figures by `hisim-tools reproduce`. The pure-ABM arm and bench alignment are fully offline; `--mock` drives the hybrid arm with a scripted client so the whole thing runs without a live backend.

---
*This file was generated by Claude Code.*
