//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `reproduce` のどのサブコマンドでも
//! 同一なので，ここ 1 箇所で組み立てる．ステップごとの集団指標の落とし方，スイープの
//! 試行 1 本ぶんの終端行，`reproduce` の条件セル・アンカー・bench 照合の書き方も
//! ここに集める．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::metrics::StepMetrics;
use crate::simulation::SimulationResult;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
/// バイナリ名 (`hisim`) と揃える．
pub const EXPERIMENT: &str = "hisim";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "mou2024";
/// 分野．網生成と初期態度に乱数を引くので `simulation` (= `master_seed` が必須)．
///
/// コア層は LLM で駆動されるが `llm-safety` ではない — 測っているのはモデルの
/// 安全性ではなく，2 階層ネットワーク上の集合的態度と動員だからである．LLM 側の
/// 同一性は `llm` ブロック ([`llm_block`]) が持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// HiSim の刻みは全エージェントを同期更新する離散タイムステップ $t$ (論文の
/// $T = 14$) そのもので，runvault の語彙では `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．集団指標はどれも母集団全体の集約なので `run`．
const SCOPE: &str = "run";

/// アンカー判定のイベント種別．コア語彙に無いので `x.<repo_id>.<name>` を使う．
pub const ANCHOR_EVENT: &str = "x.mou2024.anchor";
/// SoMoSiMu-Bench 照合のイベント種別．
pub const BENCH_EVENT: &str = "x.mou2024.bench_alignment";

/// この再現実験が対象としている論文．
///
/// どのサブコマンドも同じ主張を対象とする — `sweep` は core-ratio / ABM 種別 /
/// ネットワーク構造の感度を見るが，その存在理由は «コア層が周辺層を動かす» が
/// どの条件で成り立つかを問うことなので，同じ target に属する．論文の特定の図表
/// ではなく主張の再現を狙うので `Target::claim` を使う．
pub fn replication() -> Replication {
    Work::arxiv("2402.16333")
        .title(
            "Unveiling the Truth and Facilitating Change: Towards Agent-based \
             Large-scale Social Movement Simulation",
        )
        .year(2024)
        .source_version("published")
        .target(Target::claim(
            "core-tier-drives-mobilization",
            "A small LLM-driven core tier steers the collective attitude and mobilization of an ABM-driven periphery",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/mou2024/設計書.md")
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める (旧 `run_metadata.json` の `provider` と同じ判定)．推測しているのは
/// 分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.1` のような動くエイリアスであることが多い．
/// socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.contains("11434") || endpoint.contains("ollama") {
        "ollama"
    } else if endpoint.contains("mock") {
        "mock"
    } else {
        "openai"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // コア層のプロンプトはエージェントの profile / memory から毎回組み立てられ，
        // 固定の system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// シミュレーション 1 本
// ---------------------------------------------------------------------------

/// シミュレーション 1 本ぶんの記録 (`run` サブコマンド用)．
///
/// ステップごとの 6 指標 (`t` は時間軸なので値としては書かない) と，run 全体を
/// 1 つの値で表す `converged` / `final_step` / LLM 呼び出しの内訳を書く．
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にはしない．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    log_history(run, None, &result.metrics_history);
    run.log_metrics(
        SCOPE,
        &[
            ("converged", if result.converged { 1.0 } else { 0.0 }),
            ("final_step", result.final_step as f64),
            ("llm_calls", result.metadata.total() as f64),
            ("llm_cache_hits", result.metadata.cache_hits() as f64),
            ("llm_cache_hit_rate", result.metadata.cache_hit_rate()),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

/// メトリクス履歴をステップごとの指標として書く．
///
/// `prefix` は条件ラベル (`reproduce` は 11 条件を 1 本の run に書くので，
/// `(step, scope, name)` が衝突しないよう名前で条件を分ける)．`run` サブコマンドは
/// 条件が 1 つしかないので接頭辞なし．
pub fn log_history(run: &mut Run, prefix: Option<&str>, history: &[StepMetrics]) {
    for m in history {
        log_step(run, prefix, m);
    }
}

/// [`StepMetrics`] の 6 フィールドを 1 ステップぶんまとめて書く．
///
/// `mobilized` と `llm_actions` は個数だが，どちらも «そのステップの数» であって
/// カテゴリではないのでそのまま指標にする．
fn log_step(run: &mut Run, prefix: Option<&str>, m: &StepMetrics) {
    let name = |base: &str| match prefix {
        Some(p) => format!("{p}_{base}"),
        None => base.to_string(),
    };
    run.log_metrics_at(
        m.t as u64,
        T_UNIT,
        SCOPE,
        &[
            (name("macro_bias").as_str(), m.macro_bias),
            (name("macro_diversity").as_str(), m.macro_diversity),
            (name("mobilized").as_str(), m.mobilized as f64),
            (name("polarization").as_str(), m.polarization),
            (name("core_influence").as_str(), m.core_influence),
            (name("llm_actions").as_str(), m.llm_actions as f64),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

/// 接頭辞付きの run スコープ指標をまとめて書く．
///
/// `reproduce` の条件セル (11 条件) と bench の観測量に使う．接頭辞の付け方は
/// [`log_history`] と同じ理由 — 1 本の run に同居する条件を名前で分ける．
pub fn log_prefixed(run: &mut Run, prefix: &str, values: &[(&str, f64)]) {
    let named: Vec<(String, f64)> = values
        .iter()
        .map(|(name, v)| (format!("{prefix}_{name}"), *v))
        .collect();
    let pairs: Vec<(&str, f64)> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    run.log_metrics(SCOPE, &pairs)
        .unwrap_or_else(|e| panic!("{prefix} の run スコープ指標の記録に失敗: {e}"));
}

/// 接頭辞なしの run スコープ指標をまとめて書く．
pub fn log_scoped(run: &mut Run, values: &[(&str, f64)]) {
    run.log_metrics(SCOPE, values)
        .expect("run スコープの指標の記録に失敗");
}

// ---------------------------------------------------------------------------
// スイープの試行 (子 run の events.jsonl)
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く観測行．
///
/// 予約キーだけを持つ．数はここには書かない — 試行の最終値は下の [`TerminalEvent`]
/// が正本なので，同じ数を 2 箇所に置かない．この行が持つのは «その単位をいつ見たか»
/// という時間軸だけである．
///
/// `runvault verify --deep` は terminal の `unit_id` が observation にも現れ，
/// その最大 `t` が terminal の `t` と一致することを要求するので，観測した時刻を
/// 明示的に残す．
#[derive(Serialize)]
struct ObservationEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// `events.jsonl` に書く終端行 (旧 `sweep_summary.csv` の 1 行に対応)．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
/// 残りは自由欄．
///
/// 派生シードを `seed` ではなく `trial_seed` と呼ぶのは，`runvault.read` の
/// `sweep_events_table` が条件パラメータの列をイベント列の上に書くからである．
/// 子 run の `parameters` は基点シードを `seed` という名前で持つので，同じ名前を
/// 使うと試行ごとのシードが黙って基点シードに潰される．
#[derive(Serialize)]
struct TerminalEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    trial_seed: u64,
    final_macro_bias: f64,
    final_macro_diversity: f64,
    final_mobilized: usize,
    final_polarization: f64,
    final_core_influence: f64,
    total_llm_calls: usize,
    cache_hit_rate: f64,
}

/// 試行 1 本を `terminal` イベントとして書く (観測時刻を 1 点添えて)．
///
/// 打ち切り (`censored`) の行は `t == budget` でなければならない．ドライバは
/// 態度の変化が `tol` を下回ったら停止し，止まらなければ `steps` まで回すので，
/// 収束しなかった試行は必ず上限に達している．この不変条件は runvault が
/// `log_event` の書き込み時に検査するので，ここでは二重に持たない．
///
/// `sweep` が見るのは各試行の最終ステップだけなので，観測時刻もそこ 1 点である．
pub fn log_trial(
    run: &mut Run,
    unit_id: &str,
    trial_seed: u64,
    budget: usize,
    result: &SimulationResult,
) {
    let last = result
        .metrics_history
        .last()
        .expect("metrics_history は t=0 を含む");

    run.log_event(
        "observation",
        &ObservationEvent {
            unit_id,
            t: result.final_step as u64,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の observation の記録に失敗: {e}"));

    let event = TerminalEvent {
        unit_id,
        t: result.final_step as u64,
        t_unit: T_UNIT,
        outcome: if result.converged {
            "converged"
        } else {
            "unconverged"
        },
        censored: !result.converged,
        budget: budget as u64,
        trial_seed,
        final_macro_bias: last.macro_bias,
        final_macro_diversity: last.macro_diversity,
        final_mobilized: last.mobilized,
        final_polarization: last.polarization,
        final_core_influence: last.core_influence,
        total_llm_calls: result.metadata.total(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
    };
    run.log_event("terminal", &event)
        .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

// ---------------------------------------------------------------------------
// 条件 1 点ぶんの集約 (sweep の子 run)
// ---------------------------------------------------------------------------

/// 1 条件で回した試行 1 本の最終値．集約の材料になる．
pub struct TrialOutcome {
    /// 収束したか．
    pub converged: bool,
    /// 最終ステップの平均態度．
    pub macro_bias: f64,
    /// 最終ステップの態度分散．
    pub macro_diversity: f64,
    /// 最終ステップの動員数．
    pub mobilized: usize,
    /// 最終ステップの分極化．
    pub polarization: f64,
    /// 最終ステップのコア層平均態度．
    pub core_influence: f64,
    /// 総 LLM 呼び出し数．
    pub llm_calls: usize,
}

impl TrialOutcome {
    /// [`SimulationResult`] の最終ステップから取り出す．
    pub fn from_result(result: &SimulationResult) -> Self {
        let last = result
            .metrics_history
            .last()
            .expect("metrics_history は t=0 を含む");
        TrialOutcome {
            converged: result.converged,
            macro_bias: last.macro_bias,
            macro_diversity: last.macro_diversity,
            mobilized: last.mobilized,
            polarization: last.polarization,
            core_influence: last.core_influence,
            llm_calls: result.metadata.total(),
        }
    }
}

/// 1 条件 (core-ratio × ABM × network の 1 点) を 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行
/// ごとの `polarization` を指標にすると (`run_uid`, `step`, `scope`, `name`) が
/// 重複するので，散らばりが要る図は `events.jsonl` から組み直す．
pub fn log_condition_summary(run: &mut Run, trials: &[TrialOutcome]) {
    let n = trials.len();
    assert!(n > 0, "試行が 1 本もありません");
    let n_f = n as f64;

    let n_converged = trials.iter().filter(|t| t.converged).count();
    let mean = |f: &dyn Fn(&TrialOutcome) -> f64| trials.iter().map(f).sum::<f64>() / n_f;

    run.log_metrics(
        SCOPE,
        &[
            ("n_units", n_f),
            ("n_converged", n_converged as f64),
            ("mean_final_macro_bias", mean(&|t| t.macro_bias)),
            ("mean_final_macro_diversity", mean(&|t| t.macro_diversity)),
            ("mean_final_mobilized", mean(&|t| t.mobilized as f64)),
            ("mean_final_polarization", mean(&|t| t.polarization)),
            ("mean_final_core_influence", mean(&|t| t.core_influence)),
            ("mean_llm_calls", mean(&|t| t.llm_calls as f64)),
        ],
    )
    .expect("条件 1 点の集約の記録に失敗");
}

// ---------------------------------------------------------------------------
// reproduce のアンカー・bench 照合
// ---------------------------------------------------------------------------

/// 判定の伴うイベント 1 行を書く．
///
/// PASS / off や «整合したか» はカテゴリであって指標ではないので `events.jsonl` に
/// 置く．照合先の帯 (アンカーの `target_lo`/`target_hi`，bench の `reference`) は
/// 論文が報告した数値ではなく，この再現実装が置いた定性的なアンカーなので，出典を
/// 要求する `reference.csv` には書かない — 書くと論文の報告値と自前のアンカーが
/// 後から見分けられなくなる．
pub fn log_verdict<T: Serialize + ?Sized>(run: &mut Run, kind: &str, label: &str, payload: &T) {
    run.log_event(kind, payload)
        .unwrap_or_else(|e| panic!("{kind} の {label} の記録に失敗: {e}"));
}

// ---------------------------------------------------------------------------
// シードの派生
// ---------------------------------------------------------------------------

/// 派生シードのラベルに使う文字列ハッシュ (FNV-1a; explicit identity)．
pub fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// スイープの試行 1 本のシードを基点シードから決定的に派生させる．
///
/// `master_seed` として記録するのは `base` の方で，実際に各試行が使うシードは
/// これで作る．`(base, network, abm, core_ratio, index)` が同じなら常に同じ値を
/// 返し，どれか 1 つでも違えば別の値になる — この性質が壊れると，記録した
/// `master_seed` から run を組み直せなくなる．
pub fn sweep_trial_seed(
    base: u64,
    network: &str,
    abm: &str,
    core_ratio: f64,
    index: usize,
) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            label_hash(network),
            label_hash(abm),
            (core_ratio * 1000.0) as u64,
            index as u64,
        ],
    )
}

/// `reproduce` の 1 セル内の試行 1 本のシードを派生させる．
pub fn repro_trial_seed(base: u64, regime: &str, abm: &str, dataset: &str, index: usize) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            label_hash(regime),
            label_hash(abm),
            label_hash(dataset),
            index as u64,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::{repro_trial_seed, sweep_trial_seed};

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(
            sweep_trial_seed(42, "ba", "bc", 0.3, 2),
            sweep_trial_seed(42, "ba", "bc", 0.3, 2)
        );
        assert_eq!(
            repro_trial_seed(42, "hybrid", "bc", "metoo", 1),
            repro_trial_seed(42, "hybrid", "bc", "metoo", 1)
        );
    }

    #[test]
    fn each_coordinate_changes_the_sweep_seed() {
        let base = sweep_trial_seed(42, "ba", "bc", 0.3, 0);
        assert_ne!(base, sweep_trial_seed(43, "ba", "bc", 0.3, 0), "base");
        assert_ne!(base, sweep_trial_seed(42, "ws", "bc", 0.3, 0), "network");
        assert_ne!(base, sweep_trial_seed(42, "ba", "hk", 0.3, 0), "abm");
        assert_ne!(base, sweep_trial_seed(42, "ba", "bc", 0.4, 0), "core_ratio");
        assert_ne!(base, sweep_trial_seed(42, "ba", "bc", 0.3, 1), "index");
    }

    #[test]
    fn each_coordinate_changes_the_repro_seed() {
        let base = repro_trial_seed(42, "hybrid", "bc", "metoo", 0);
        assert_ne!(base, repro_trial_seed(43, "hybrid", "bc", "metoo", 0));
        assert_ne!(base, repro_trial_seed(42, "pure-abm", "bc", "metoo", 0));
        assert_ne!(base, repro_trial_seed(42, "hybrid", "hk", "metoo", 0));
        assert_ne!(base, repro_trial_seed(42, "hybrid", "bc", "roe", 0));
        assert_ne!(base, repro_trial_seed(42, "hybrid", "bc", "metoo", 1));
    }

    #[test]
    fn one_condition_gives_distinct_seeds_across_trials() {
        let seeds: std::collections::BTreeSet<u64> = (0..64)
            .map(|i| sweep_trial_seed(42, "ba", "bc", 0.3, i))
            .collect();
        assert_eq!(seeds.len(), 64, "同一条件の試行でシードが衝突した");
    }

    /// 具体値を固定する．
    ///
    /// ここが変わるのは socsim の `derive_seed` が変わったときで，そのときは
    /// 過去の run と結果を比較できなくなっている．Cargo.lock が socsim の commit を
    /// 固定しているので，この値は依存を上げたときにだけ動く．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(
            sweep_trial_seed(42, "ba", "bc", 0.0, 0),
            8_599_770_598_032_394_349
        );
        assert_eq!(
            repro_trial_seed(42, "pure-abm", "bc", "metoo", 0),
            6_808_432_238_648_355_826
        );
    }
}
