<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

**English** | [日本語](README.ja.md)

# HiSim: Towards Agent-based Large-scale Social Movement Simulation — Mou et al. (2024)

A reimplementation of the **HiSim** framework of Mou, Wei & Huang (2024), "Unveiling the Truth and Facilitating Change: Towards Agent-based Large-scale Social Movement Simulation" (Findings of ACL 2024; arXiv:2402.16333). HiSim predicts how collective opinion evolves on social media after a trigger event by splitting users into **two tiers**: a small set of active, influential **core** users are driven by an LLM (profile / memory → an LLM picks an action: post / retweet / reply / like / do-nothing → stance/sentiment post-processing updates the attitude), while the silent majority of **ordinary** users are driven by a deterministic **agent-based opinion-dynamics model** (Bounded Confidence, HK, Social Judgement, or Lorenz). The coupling is **one-way**: core users influence ordinary users, but the reverse is treated as negligible. Both tiers sit on a single social network (Barabási–Albert by default), and every agent holds a continuous attitude `a ∈ [-1, 1]`.

The deterministic [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) core handles network generation, tier assignment, the ordinary-tier ABM updates, the scheduler and the metrics; the non-deterministic LLM layer is confined to a single mechanism and pseudo-determinised via the `socsim-llm` crate (prompt→response cache + `temperature=0` + fixed seed). With `--core-ratio 0.0` there are **no LLM calls** at all — a fully deterministic pure-ABM baseline.

## Two-layer determinism (read this first)

LLM output is **outside** socsim's bit-reproducibility. The design therefore splits into two layers:

- **Deterministic socsim core** — network generation (BA / WS / ER), tier assignment (top-degree nodes become core), the ordinary-tier ABM opinion dynamics (synchronous update from a start-of-step attitude snapshot), the scheduler and the metrics. Given a seed this reproduces bit-for-bit.
- **Non-deterministic LLM layer** — the core tier's action choice. Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed seed. The provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses, so a rerun is free and stable. Each run writes `run_metadata.json` recording the provider, model, endpoint, temperature, seed, core-ratio and cache-hit rate. Because the local default model (`llama3.2:latest`) differs from the paper's GPT-3.5, reproduction targets are **qualitative** (trends and signs: the hybrid corrects the pure-ABM trend, BC/HK converge toward consensus, SJ/Lorenz polarize, BA amplifies core influence), not the paper's exact numbers.

## The two-tier hybrid

HiSim's contribution is reconciling **scale** (millions of users) with **fidelity** (rich LLM behaviour). Running thousands of LLMs is infeasible, so only the influential minority — the **core** — is LLM-driven, while the silent majority — the **ordinary** tier — uses a cheap deterministic ABM. This mirrors the Pareto distribution of social-media engagement (a few active users generate most content). Calibration follows the paper: tune ABM parameters on the pure-ABM (`--core-ratio 0.0`) path, then apply them to the hybrid — avoiding hundreds of LLM calls during tuning.

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with the Ollama+OpenAI backends)
cargo build --release

# === Pure-ABM baseline (no LLM required) ===
cargo run --release -- run \
    --dataset metoo --abm bc --core-ratio 0.0 \
    --n-agents 1000 --steps 14 --network ba --seed 42

# === Hybrid (LLM core + ABM periphery) — needs a local Ollama ===
#   ollama pull llama3.2:latest
OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=llama3.2:latest \
cargo run --release -- run \
    --dataset metoo --abm bc --core-ratio 0.3 \
    --n-agents 1000 --steps 14 --network ba \
    --llm-temperature 0 --seed 42
# Rerunning with the same arguments → 100% cache hits (no LLM calls)

# === Sensitivity sweep (core-ratio × ABM × network) ===
cargo run --release -- sweep \
    --core-ratio-min 0.0 --core-ratio-max 0.5 --core-ratio-step 0.1 \
    --abm-values bc,hk,sj,lorenz --network-values ba --runs 10 --seed 42

# === Reproduce the paper (Table 2/3 + SoMoSiMu-Bench) ===
#   Hybrid vs pure-ABM contrast + bench alignment; --mock = fully offline (no live LLM)
cargo run --release -- reproduce --mock --seed 42

# === External-LLM stance annotation (opt-in; needs a live backend) ===
#   The default keyword classifier is bit-identical; `--stance-annotator llm`
#   asks the core LLM to classify each post's stance on a 5-point scale.
OLLAMA_MODEL=llama3.2:latest cargo run --release -- run \
    --core-ratio 0.3 --abm bc --stance-annotator llm --seed 42

# === Visualization ===
uv sync
uv run hisim-tools visualize
uv run hisim-tools visualize-sweep
uv run hisim-tools show-experiment-settings --results-dir results/latest
uv run hisim-tools reproduce --run --mock          # reproduce report + figures, offline

# === Offline (no live LLM) smoke: hybrid path via a scripted mock client ===
cargo run --release --example mock_smoke -- results
```

## Output

Each `run` writes to `results/{timestamp}/` (with a `latest` symlink):

- `metrics.csv` — long-format `t, metric, value` for `macro_bias` (mean attitude), `macro_diversity` (variance), `mobilized` (count past threshold), `polarization` (bimodality), `core_influence` (core-tier mean attitude), `llm_actions`.
- `config.json` — the resolved run configuration.
- `run_metadata.json` — LLM provider / model / endpoint / temperature / seed / core-ratio / cache-hit rate.

Each `sweep` writes `results/{timestamp}_sweep/` with `sweep_summary.csv` and `sweep_config.json`.

Each `reproduce` writes `results/reproduce_{timestamp}/` with `reproduce_summary.json` (the Table 3 hybrid-vs-pure-ABM matrix, the SoMoSiMu-Bench alignment, and observed-vs-paper anchors with PASS/off bands), per-condition `metrics_<label>.csv`, and — via `hisim-tools reproduce` — `figures/{table3_hybrid_vs_pureabm,bench_alignment,mobilization_curves}.png`. The SoMoSiMu-Bench reference is a **calibrated synthetic** curve, not the raw benchmark dataset (see [Architecture](docs/architecture.md)).

## Documentation

- [Architecture](docs/architecture.md) — repository layout, the `HiSimWorld` two-tier state, the four mechanisms, the ABM math, stance annotation, SoMoSiMu-Bench alignment and `reproduce`.
- [CLI reference](docs/cli.md) — every `run` / `sweep` / `reproduce` flag.
- [Use cases](docs/usecases.md) — pure-ABM baseline, hybrid, sensitivity sweeps, paper reproduction.
- [Visualization](docs/visualization.md) — the Python tools.

## References

- Mou, X., Wei, Z., & Huang, X. (2024). Unveiling the Truth and Facilitating Change: Towards Agent-based Large-scale Social Movement Simulation. *Findings of ACL 2024*, 4789–4809.
- Deffuant, G., et al. (2000). Mixing Beliefs among Interacting Agents. *Advances in Complex Systems* (Bounded Confidence).
- Hegselmann, R., & Krause, U. (2002). Opinion Dynamics and Bounded Confidence (HK).
- Lorenz, J., et al. (2021). A Model of Opinion Dynamics with Assimilation, Reinforcement, and Polarization.

## License

MIT — see [LICENSE](LICENSE).

---
*This file was generated by Claude Code.*
