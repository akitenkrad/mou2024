**English** | [日本語](cli.ja.md)

# CLI reference

The `hisim` binary has three subcommands: `run`, `sweep`, `reproduce`.

## `run` — single configuration

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | `metoo` | dataset / trigger-event context (`metoo` / `roe` / `blm`) |
| `--abm` | `bc` | ordinary-tier opinion-dynamics model (`bc` / `hk` / `sj` / `lorenz`) |
| `--core-ratio` | `0.3` | fraction of LLM-driven core agents ∈ [0,1]; `0.0` = pure ABM (no LLM) |
| `--n-agents` | `1000` | number of agents N |
| `--steps` | `14` | timesteps T |
| `--network` | `ba` | network generator (`ba` / `ws` / `er`) |
| `--ba-m` | `4` | BA edges per new node m |
| `--ws-k` / `--ws-beta` | `6` / `0.1` | WS neighbours k / rewiring β |
| `--er-p` | `0.02` | ER edge probability p |
| `--alpha` / `--epsilon` | `0.3` / `0.4` | ABM assimilation α / confidence bound ε |
| `--mobilization-threshold` | `0.5` | `|attitude| ≥ this` counts as mobilized |
| `--llm-budget` | `5000` | max LLM calls per run (overflow → do-nothing) |
| `--seed` | random | RNG seed (governs the socsim core only) |
| `--llm-temperature` | `0.0` | LLM generation temperature |
| `--llm-seed` | `0` | LLM backend seed |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache file |
| `--stance-annotator` | `deterministic` | core-post stance → attitude mapping: `deterministic` (keyword classifier; the default, bit-identical to prior behaviour, no extra LLM calls) or `llm` (external-LLM stance annotation — asks the core LLM to classify each post's stance on a 5-point scale; needs a live backend) |
| `--output-dir` | `results` | runvault results root |

Output goes to a runvault run directory. The run directory *is* the output directory, so no timestamped subdirectory and no `latest` symlink are created. Ask `runvault` for the most recent finished run:

```bash
runvault path --experiment hisim --latest --subcommand run
```

```
results/
└── hisim/                                          ← experiment
    ├── latest_finished -> run_20260405_153000_...   ← the last run that finished
    ├── run_20260405_153000_9f2c41ab_3b1d/           ← <subcommand>_<time>_<cfg8>_<exec4>
    │   ├── run.json                                 ← metadata (git commit / env / LLM / paper)
    │   ├── config.json                              ← envelope; the conditions sit under ["parameters"]
    │   ├── metrics.csv                              ← long form (step / step_unit / scope / name / value)
    │   ├── status.json                              ← outcome and duration
    │   └── manifest.csv                             ← hashes of artifacts/ and logs/
    └── figures/                                     ← what the plotting scripts write (outside the run)
        └── run_20260405_153000_9f2c41ab_3b1d/
            └── metrics_timeseries.png
```

`metrics.csv` is long form, one value per row. The six per-step metrics (`macro_bias` / `macro_diversity` / `mobilized` / `polarization` / `core_influence` / `llm_actions`) carry a `step` with `step_unit=step`; `converged` (0.0 / 1.0) / `final_step` / `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` describe the whole run with one number each and carry no step. The LLM model / provider / temperature live in the `llm` block of `run.json` (no `run_metadata.json` is written).

## `sweep` — sensitivity analysis

Sweeps `core-ratio × abm × network`, running `--runs` independent trials per condition.

| Flag | Default | Meaning |
|------|---------|---------|
| `--core-ratio-min/max/step` | `0.0` / `0.5` / `0.1` | core-ratio sweep range |
| `--abm-values` | `bc,hk,sj,lorenz` | comma-separated ABM models |
| `--network-values` | `ba` | comma-separated networks |
| `--n-agents` | `1000` | agents per run |
| `--steps` | `14` | timesteps |
| `--runs` | `10` | independent trials per condition |
| `--llm-budget` | `5000` | LLM call cap |
| `--seed` | `42` | base seed (each trial derives an independent seed) |
| `--llm-temperature` / `--llm-seed` | `0.0` / `0` | LLM settings |
| `--cache-path` | `.llm_cache/cache.json` | shared cache (raises hit rate across the sweep) |
| `--output-dir` | `results` | runvault results root |

`--core-ratio 0.0` runs are pure ABM and make no LLM calls — cheap enough to raise `--runs` to 30.

A sweep is recorded as one parent run plus one child run per condition. The children take the subcommand name `sweep-point`, sit beside the parent in the experiment directory rather than under it, and point at the parent through `lineage.parent_run_uid`. No per-trial summary CSV is written — the same values are in the children's `events.jsonl`.

```
results/
└── hisim/
    ├── sweep_20260405_160827_48d033b7_ee20/         ← parent; its parameters are the grid
    │   ├── run.json                                  ← carries lineage.sweep_id; rng.master_seed is null
    │   └── config.json
    ├── sweep-point_20260405_160828_174916dd_955d/   ← child = one condition's trials
    │   ├── config.json                               ← that condition (network / abm / core_ratio)
    │   ├── metrics.csv                               ← the condition's aggregate (n_units / n_converged / mean_final_*)
    │   └── events.jsonl                              ← one trial = one terminal line
    └── ...
```

`runvault path --experiment hisim --latest --subcommand sweep` prints the parent. `hisim-tools visualize-sweep` takes that parent, collects the children's `terminal` lines and rebuilds the familiar one-row-per-trial table.

## `reproduce` — Table 2/3 + SoMoSiMu-Bench

Reproduces the paper's headline results in one shot: the **Table 3** contrast of the hybrid regime (LLM core + ABM periphery) against the pure-ABM baseline (`--core-ratio 0`) across all four ordinary-tier models, and the **Table 2** SoMoSiMu-Bench movement-metric alignment per dataset. Eleven conditions (the eight Table 3 cells plus the three bench movements) share one run, so the representative run's per-step series and the per-condition trial means are named `<condition>_<metric>` in `metrics.csv` (e.g. `hybrid_bc_polarization`, `pureabm_bc_mean_final_bias`). The anchor and bench verdicts are categories rather than numbers, so they go to `events.jsonl` as `x.mou2024.anchor` / `x.mou2024.bench_alignment` (the observed values themselves are also run-scope metrics).

| Flag | Default | Meaning |
|------|---------|---------|
| `--datasets` | `metoo,roe,blm` | comma-separated movements to align against the bench |
| `--abm-values` | `bc,hk,sj,lorenz` | comma-separated ordinary-tier models for the Table 3 matrix |
| `--core-ratio` | `0.3` | core ratio for the hybrid arm (the pure-ABM arm is always `0.0`) |
| `--n-agents` | `600` | agents per run |
| `--steps` | `14` | timesteps |
| `--runs` | `5` | independent trials per condition |
| `--network` | `ba` | network generator |
| `--seed` | `42` | base seed (each condition/trial derives an independent seed) |
| `--mock` | off | drive the hybrid arm with a deterministic scripted client (no live LLM); the pure-ABM arm never needs it |
| `--stance-annotator` | `deterministic` | stance → attitude mapping (`deterministic` / `llm`) |
| `--llm-temperature` / `--llm-seed` / `--cache-path` | `0.0` / `0` / `.llm_cache/cache.json` | live-LLM settings (hybrid arm only) |
| `--quick` | off | shrink N / runs / steps for a fast smoke check |
| `--output-dir` | `results` | runvault results root |

The pure-ABM arm and the bench alignment are fully offline (`--core-ratio 0` makes zero LLM calls); add `--mock` to also run the hybrid arm without a live backend. The SoMoSiMu-Bench reference is a **synthetic** curve (the raw benchmark dataset is not bundled). It is an anchor this replication chose rather than a value the paper reports, so it does not go into `reference.csv`, which demands a source — see [architecture](architecture.md) for the honest split.

---
*This file was generated by Claude Code.*
