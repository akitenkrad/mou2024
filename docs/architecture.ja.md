[English](architecture.md) | **日本語**

# アーキテクチャ

## リポジトリ構成

```
mou2024/
├── Cargo.toml                  # Rust workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (mou2024-workspace)
├── simulation/                 # Rust crate hisim-simulation (bin hisim)
│   ├── src/
│   │   ├── main.rs             # CLI (clap: run / sweep / reproduce)
│   │   ├── config.rs           # Config, AbmModel, NetworkKind, LlmSettings
│   │   ├── world.rs            # HiSimWorld (WorldState), Tier, CoreState
│   │   ├── abm.rs              # BC / HK / SJ / Lorenz の f_update + f_message
│   │   ├── prompts.rs          # コア層の行動選択プロンプト
│   │   ├── parse.rs            # ACTION / MESSAGE のパース
│   │   ├── mechanisms.rs       # 4 メカニズム (Environment / Decision / Mobilization / Aggregate)
│   │   ├── llm.rs              # Ollama→OpenAI フォールバック + キャッシュ + stance 分類器 + StanceMode (決定論 / 外部 LLM)
│   │   ├── bench.rs            # SoMoSiMu-Bench 運動指標 + 較正済み合成参照 + 照合
│   │   ├── reproduce_mock.rs   # `reproduce --mock` 用のオフライン scripted client
│   │   ├── simulation.rs       # init_world + run ドライバ + 出力 writer
│   │   └── metrics.rs          # macro_bias / diversity / mobilized / polarization / core_influence
│   ├── examples/mock_smoke.rs  # オフライン (LLM 不要) ハイブリッドスモーク
│   └── tests/integration_test.rs
├── tools/src/hisim_tools/      # Python package hisim-tools
│   ├── cli.py                  # サブコマンドディスパッチャ
│   ├── visualize.py            # 態度 / 動員 / 多様性-分極化 / コア影響
│   ├── visualize_sweep.py      # 比率 / モデル / ネットワーク依存
│   ├── show_experiment_settings.py
│   └── reproduce_paper.py      # Table 2/3 + SoMoSiMu-Bench レポート・図
└── results/                    # 実行時生成 (gitignore 対象)
```

## `HiSimWorld` — 2 階層の世界状態

```rust
pub struct HiSimWorld {
    pub clock: SimClock,
    pub attitude: BTreeMap<AgentId, f64>,   // a ∈ [-1, 1]，全エージェント (ソート済みキー)
    pub tier: BTreeMap<AgentId, Tier>,      // Core | Ordinary
    pub core: BTreeMap<AgentId, CoreState>, // profile + memory (コア層のみ)
    pub network: SocialNetwork,             // BA (既定) / WS / ER
    pub current_event: String,              // 当該ステップの外部イベント
    pub macro_bias: f64,
    pub macro_diversity: f64,
    pub mobilized: usize,
}
```

`agent_ids()` は `BTreeMap` のソート済みキーを返し，socsim コアの決定論的反復を保証する．階層割当は次数上位 `ceil(core_ratio · N)` 体をコア層にする (スケールフリー網では高次数 = 影響力大)．

## 4 メカニズム (フェーズごとに 1 つ)

| Mechanism | Phase | 役割 |
|-----------|-------|------|
| `EnvironmentMechanism`   | Environment | 当該ステップのトリガーニュース・外部イベントを `current_event` に提示 (決定論的)． |
| `DecisionMechanism`      | Decision    | **コア層** が LLM を呼び (唯一の LLM 呼び出し点)，行動をパースし，発信メッセージを stance 分類し，自身の態度を更新して発信態度を記録する．**一般層** はここでは何もしない (その `f_selection` = ネットワーク隣接の収集は次フェーズ)． |
| `MobilizationMechanism`  | Interaction | 一般層を同期更新: ステップ開始時の態度をスナップショットし，各エージェントのメッセージ集合 `M_i` を隣接の `f_message(a_j)=a_j` + フォロー先コアの発信 (一方向結合) から構成し，`Δa = f_update(a_i, M_i)` を計算して一括書き戻す． |
| `AggregateMechanism`     | Reward      | `macro_bias / diversity / mobilized / polarization` を計算・記録し，収束で `request_stop` (T はエンジンのクロックが担保)． |

LLM 呼び出しは `DecisionMechanism` に閉じ込める．`--core-ratio 0.0` ではコア層が空なので LLM 呼び出しは一切起きず，全体が bit 決定論的になる．

## ABM レイヤ (`abm.rs`)

統一定式化 `f_update / f_selection / f_message` (Chuang & Rogers 2023; 設計 §6) に従う:

- `f_message(a_j) = a_j` — メッセージは送信者の態度をバイアスなく伝える．
- `f_update(a_i, M_i)` は **差分** `Δa` を返す; 呼び出し側が `a_i + Δa` を `[-1, 1]` にクランプして同期書き戻す．

| モデル | 規則 | 定性挙動 |
|--------|------|---------|
| **BC** (Deffuant) | 信頼境界 ε 内のソースについて `Δa = α·(m_j − a_i)` を平均 | 合意形成 |
| **HK** | 信頼境界内ソース (自身含む) の平均へ α で移動 | 合意形成 / クラスタ |
| **SJ** (Social Judgement) | 受容域は同化，拒否域は反発 | 二極化 |
| **Lorenz** | 同化 + 強化 + 分極化 (極端を増幅) | 二極化 |

一般層の数式は hegselmann2005 の同期有界信頼更新を踏襲する．

## 2 つの RNG ストリーム

```rust
const RNG_WORLD_INIT: u64 = 0; // 網生成・態度初期化・階層割当
const RNG_ENGINE: u64    = 1;  // RandomActivationScheduler + 一般 ABM の確率処理
```

いずれも root seed から `derive_seed` で派生．スケジューラは `RandomActivationScheduler` (一般層は同期更新なので順序は結果に影響しない)．

## stance 注釈 (`llm.rs`, `StanceMode`)

コア post のテキストは `--stance-annotator` で選ぶ 2 経路のいずれかで態度 `[-1, 1]` へ写像する:

- **`deterministic`** (既定) — 軽量キーワード分類器 (`classify_stance`; 論文の stance/sentiment 注釈の代替)．同一テキスト → 同一スコアで，**追加 LLM 呼び出し無し**．従来挙動とビット等価なので，決定論的コアの再現性とキャッシュ整合を保つ．
- **`llm`** — 外部 LLM 注釈経路．コア LLM に post の stance を 5 段階で答えさせ (`stance_annotation_prompt`)，整数応答を `[-1, 1]` へ写像する (`classify_stance_llm`; 整数が読めなければ決定論的分類器にフォールバック)．これらの注釈呼び出しはメタデータ・予算に計上され，キャッシュ対象なので `temperature=0` + 固定 seed + プロンプトキャッシュで再実行時の擬似決定論を保つ．

## SoMoSiMu-Bench 照合 (`bench.rs`)

`bench.rs` は HiSim 出力を SoMoSiMu-Bench 流の **運動指標** (`MovementMetrics`: 動員ピーク / ピーク時刻 / 最終動員 / 最終 bias / 最終分極化 / 持続比) へ写像し，運動別の参照と比較して指標別の許容帯を持つ `BenchComparison` を作る．

**実 vs 較正 (正直な区分):** *観測*指標はすべてシミュレータの実出力である．*参照*は **較正済み合成**曲線 (`reference_curve`; `source = "calibrated-synthetic"` でタグ付け) で，各運動の論文の定性記述 (例: RoeOverturned は #MeToo より分極化が強い) に手で合わせたものであり ground-truth ではない．生 SoMoSiMu-Bench データは同梱しない (取得・ライセンスは原著者の管理下)．入手できれば `BenchReference` を CSV ローダに差し替えるだけで同じ照合経路が使える．比較は数値完全一致ではなく定性傾向 (符号 / 順序 / 帯) の整合を見る (`reproduce` のアンカーと同じ哲学)．

## `reproduce` — Table 2/3 一括

`hisim reproduce` は，先頭データセットについて全ての一般層モデルを **ハイブリッド** (LLM コア + ABM 周辺) と **純 ABM** (`core-ratio 0`) の両レジームで回し，最終 Bias / Diversity / Polarization / Mobilization と動員の伸びを集計する (Table 3)．次に各データセットの純 ABM 運動ダイナミクスを較正済み bench 参照に整合させる (Table 2)．結果は観測 vs 論文のアンカー (PASS/off 帯 — 例: *純 ABM は LLM 0 呼び出し*・*Lorenz/SJ は BC より分極*・*BC/HK は合意*・*LLM コアが動員を増幅*) と bench 照合を含む `reproduce_summary.json` で，`hisim-tools reproduce` がレポートと 3 つの図として可視化する．純 ABM 条件と bench 照合は完全オフラインで，`--mock` はハイブリッド条件を scripted client で駆動するためライブバックエンド無しで全体が走る．

---
*This file was generated by Claude Code.*
