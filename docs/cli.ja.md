[English](cli.md) | **日本語**

# CLI リファレンス

`hisim` バイナリは 3 つのサブコマンドを持つ: `run`・`sweep`・`reproduce`．

## `run` — 単一設定

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--dataset` | `metoo` | データセット / トリガーイベント文脈 (`metoo` / `roe` / `blm`) |
| `--abm` | `bc` | 一般層の意見力学モデル (`bc` / `hk` / `sj` / `lorenz`) |
| `--core-ratio` | `0.3` | LLM 駆動コアの比率 ∈ [0,1]; `0.0` = 純粋 ABM (LLM なし) |
| `--n-agents` | `1000` | エージェント数 N |
| `--steps` | `14` | タイムステップ数 T |
| `--network` | `ba` | ネットワーク生成器 (`ba` / `ws` / `er`) |
| `--ba-m` | `4` | BA の新規ノードあたり結合数 m |
| `--ws-k` / `--ws-beta` | `6` / `0.1` | WS の近傍数 k / 張り替え β |
| `--er-p` | `0.02` | ER の辺確率 p |
| `--alpha` / `--epsilon` | `0.3` / `0.4` | ABM 同化率 α / 信頼境界 ε |
| `--mobilization-threshold` | `0.5` | `|態度| ≥ この値` で動員とみなす |
| `--llm-budget` | `5000` | 1 実行の最大 LLM 呼び出し数 (超過 → do-nothing) |
| `--seed` | ランダム | RNG シード (socsim コアのみ支配) |
| `--llm-temperature` | `0.0` | LLM 生成温度 |
| `--llm-seed` | `0` | LLM バックエンド seed |
| `--cache-path` | `.llm_cache/cache.json` | プロンプト→応答キャッシュ |
| `--stance-annotator` | `deterministic` | コア post の stance → 態度 写像: `deterministic` (キーワード分類器; 既定・従来挙動とビット等価・追加 LLM 呼び出し無し) または `llm` (外部 LLM stance 注釈 — コア LLM に各 post の stance を 5 段階で答えさせる; ライブバックエンドが必要) |
| `--output-dir` | `results` | runvault の results ルート |

出力は runvault の run ディレクトリへ．run ディレクトリが出力先そのものなので，タイムスタンプ付きサブディレクトリも `latest` symlink も作らない．直近の完了 run のパスは `runvault` に聞く:

```bash
runvault path --experiment hisim --latest --subcommand run
```

```
results/
└── hisim/                                          ← experiment
    ├── latest_finished -> run_20260405_153000_...   ← 最後に完了した run
    ├── run_20260405_153000_9f2c41ab_3b1d/           ← <subcommand>_<時刻>_<cfg8>_<exec4>
    │   ├── run.json                                 ← メタデータ (git commit / 環境 / LLM / 論文情報)
    │   ├── config.json                              ← 封筒．実験条件は ["parameters"] の下
    │   ├── metrics.csv                              ← long 形式 (step / step_unit / scope / name / value)
    │   ├── status.json                              ← 終了状態と所要時間
    │   └── manifest.csv                             ← artifacts/ と logs/ のハッシュ
    └── figures/                                     ← 可視化スクリプトの出力 (run の外)
        └── run_20260405_153000_9f2c41ab_3b1d/
            └── metrics_timeseries.png
```

`metrics.csv` は 1 行 1 値の long 形式．ステップごとの 6 指標 (`macro_bias` / `macro_diversity` / `mobilized` / `polarization` / `core_influence` / `llm_actions`) は `step_unit=step` の `step` を持ち，run 全体を 1 つの値で表す `converged` (0.0 / 1.0) / `final_step` / `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` は `scope=run` で `step` を持たない．LLM のモデル・provider・温度は `run.json` の `llm` ブロックにある (`run_metadata.json` は書かれない)．

## `sweep` — 感度分析

`core-ratio × abm × network` を走査し，条件ごとに `--runs` 回の独立試行を行う．

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--core-ratio-min/max/step` | `0.0` / `0.5` / `0.1` | コア比率スイープ範囲 |
| `--abm-values` | `bc,hk,sj,lorenz` | カンマ区切り ABM モデル |
| `--network-values` | `ba` | カンマ区切りネットワーク |
| `--n-agents` | `1000` | 各実行のエージェント数 |
| `--steps` | `14` | タイムステップ数 |
| `--runs` | `10` | 条件ごとの独立試行数 |
| `--llm-budget` | `5000` | LLM 呼び出し上限 |
| `--seed` | `42` | 基点 seed (各試行は独立 seed を派生) |
| `--llm-temperature` / `--llm-seed` | `0.0` / `0` | LLM 設定 |
| `--cache-path` | `.llm_cache/cache.json` | 共有キャッシュ (スイープ全体でヒット率向上) |
| `--output-dir` | `results` | runvault の results ルート |

`--core-ratio 0.0` の実行は純粋 ABM で LLM を呼ばないため，`--runs` を 30 まで増やせる．

sweep は「親 run 1 本 + 条件ごとの子 run」として記録される．子はサブコマンド名 `sweep-point` を名乗り，親の下ではなく experiment ディレクトリの兄弟として並び，`lineage.parent_run_uid` で親を指す．1 行 1 試行のサマリ CSV は書かない (同じ値は子の `events.jsonl` にある)．

```
results/
└── hisim/
    ├── sweep_20260405_160827_48d033b7_ee20/         ← 親．parameters が格子の定義
    │   ├── run.json                                  ← lineage.sweep_id を持つ．rng.master_seed は null
    │   └── config.json
    ├── sweep-point_20260405_160828_174916dd_955d/   ← 子 = 1 条件の試行群
    │   ├── config.json                               ← その条件 (network / abm / core_ratio)
    │   ├── metrics.csv                               ← 条件の集約 (n_units / n_converged / mean_final_*)
    │   └── events.jsonl                              ← 試行 1 本 = terminal 行 1 本
    └── ...
```

親のパスは `runvault path --experiment hisim --latest --subcommand sweep` で取れる．`hisim-tools visualize-sweep` はこの親を受け取り，子 run の `terminal` 行を集めて従来のサマリ表 (1 行 1 試行) を組み直す．

## `reproduce` — Table 2/3 + SoMoSiMu-Bench

論文の見出し的結果を一括再現する: **Table 3** のハイブリッド (LLM コア + ABM 周辺) vs 純 ABM (`--core-ratio 0`) を 4 つの一般層モデル全てで対比し，**Table 2** の SoMoSiMu-Bench 運動指標照合をデータセット別に行う．11 条件 (Table 3 の 8 セル + bench の 3 運動) が 1 本の run に同居するので，条件ごとの代表 run のステップ別系列と試行平均は `<条件ラベル>_<指標名>` (例 `hybrid_bc_polarization` / `pureabm_bc_mean_final_bias`) という名前で `metrics.csv` に入る．論文知見アンカーと bench の整合判定は数ではなくカテゴリなので `events.jsonl` の `x.mou2024.anchor` / `x.mou2024.bench_alignment` へ書く (観測値そのものは run スコープ指標にもある)．

| フラグ | 既定 | 意味 |
|--------|------|------|
| `--datasets` | `metoo,roe,blm` | bench 照合する運動 (カンマ区切り) |
| `--abm-values` | `bc,hk,sj,lorenz` | Table 3 行列の一般層モデル (カンマ区切り) |
| `--core-ratio` | `0.3` | ハイブリッド条件のコア比率 (純 ABM 条件は常に `0.0`) |
| `--n-agents` | `600` | 各実行のエージェント数 |
| `--steps` | `14` | タイムステップ数 |
| `--runs` | `5` | 条件ごとの独立試行数 |
| `--network` | `ba` | ネットワーク生成器 |
| `--seed` | `42` | 基点 seed (各条件・試行は独立 seed を派生) |
| `--mock` | off | ハイブリッド条件を決定論的 scripted client で駆動 (ライブ LLM 不要; 純 ABM 条件は不要) |
| `--stance-annotator` | `deterministic` | stance → 態度 写像 (`deterministic` / `llm`) |
| `--llm-temperature` / `--llm-seed` / `--cache-path` | `0.0` / `0` / `.llm_cache/cache.json` | ライブ LLM 設定 (ハイブリッド条件のみ) |
| `--quick` | off | N・runs・steps を縮小した高速スモーク |
| `--output-dir` | `results` | runvault の results ルート |

純 ABM 条件と bench 照合は完全オフライン (`--core-ratio 0` は LLM を一切呼ばない)．ハイブリッド条件もライブバックエンド無しで走らせるには `--mock` を付ける．SoMoSiMu-Bench 参照は **合成**曲線である (生ベンチマークデータは同梱しない)．論文が報告した値ではなくこの再現実装が置いたアンカーなので，出典を要求する `reference.csv` には書かない — 正直な区分は [architecture](architecture.ja.md) を参照．
