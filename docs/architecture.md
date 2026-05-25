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
│   │   ├── llm.rs              # Ollama→OpenAI fallback + cache + stance classifier
│   │   ├── simulation.rs       # init_world + run driver + output writers
│   │   └── metrics.rs          # macro_bias / diversity / mobilized / polarization / core_influence
│   ├── examples/mock_smoke.rs  # offline (no live LLM) hybrid smoke
│   └── tests/integration_test.rs
├── tools/src/hisim_tools/      # Python package hisim-tools
│   ├── cli.py                  # subcommand dispatcher
│   ├── visualize.py            # attitude / mobilization / diversity-polarization / core-influence
│   ├── visualize_sweep.py      # ratio / model / network dependence
│   ├── show_experiment_settings.py
│   └── reproduce_paper.py      # Phase 3 stub
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

---
*This file was generated by Claude Code.*
