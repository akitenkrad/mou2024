//! Mou et al. (2024) "Unveiling the Truth and Facilitating Change: Towards
//! Agent-based Large-scale Social Movement Simulation" (HiSim) — 再現実験の CLI
//! エントリポイント．
//!
//! `run`       : 単一設定で 2 階層ハイブリッド (LLM コア + ABM 周辺) を実行する．
//!               `--core-ratio 0.0` なら純粋 ABM (LLM 呼び出し無し)．
//! `sweep`     : コア比率 × ABM 種別 × ネットワーク構造 を走査する．親 run 1 本と，
//!               条件 1 点ごとの子 run (`sweep-point`) に分ける．
//! `reproduce` : 論文 Table 2/3 の見出し的知見 (ハイブリッド vs 純 ABM) と
//!               SoMoSiMu-Bench 照合を一括再現する．
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも
//! `latest` シンボリックリンクもこちらでは作らず，`Run::start` が決めた run
//! ディレクトリへ書く．

use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::rc::Rc;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions, Stage};
use serde::Serialize;

use hisim_simulation::bench::{compare_to_bench, reference_curve, MovementMetrics};
use hisim_simulation::config::{
    parse_abm, parse_network, parse_stance_mode, AbmModel, AbmParams, Config, LlmSettings,
    NetworkConfig, NetworkKind, StanceMode,
};
use hisim_simulation::llm::{build_live_client, HiSimClient};
use hisim_simulation::mechanisms::{no_observer, DecisionObserver};
use hisim_simulation::metrics::StepMetrics;
use hisim_simulation::record::{self, ANCHOR_EVENT, BENCH_EVENT, DOMAIN, EXPERIMENT, REPO_ID};
use hisim_simulation::reproduce_mock::build_reproduce_client;
use hisim_simulation::simulation::{run_with_client_observed, SimulationResult};

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "hisim",
    about = "Mou et al. (2024) HiSim: 大規模社会運動シミュレーション (2 階層ハイブリッド) — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で 2 階層ハイブリッド (LLM コア + ABM 周辺) を実行する．
    Run(RunArgs),
    /// コア比率 × ABM 種別 × ネットワーク構造 を走査し最終指標を集計する．
    Sweep(SweepArgs),
    /// 論文 Table 2/3 の見出し的知見 (ハイブリッド vs 純 ABM) + SoMoSiMu-Bench
    /// 照合を一括再現し reproduce_summary.json に集計する．
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// データセット (metoo / roe / blm)．
    #[arg(long, default_value = "metoo")]
    dataset: String,

    /// 周辺 ABM 種別 (bc / hk / sj / lorenz)．
    #[arg(long, default_value = "bc")]
    abm: String,

    /// コア層 (LLM 駆動) の比率 ∈ [0,1]．0.0 = 純粋 ABM．
    #[arg(long, default_value_t = 0.3)]
    core_ratio: f64,

    /// エージェント数 N．
    #[arg(long, default_value_t = 1000)]
    n_agents: usize,

    /// タイムステップ数 T．
    #[arg(long, default_value_t = 14)]
    steps: usize,

    /// ネットワーク構造 (ba / ws / er)．
    #[arg(long, default_value = "ba")]
    network: String,

    /// BA の結合数 m．
    #[arg(long, default_value_t = 4)]
    ba_m: usize,

    /// WS の近傍数 k．
    #[arg(long, default_value_t = 6)]
    ws_k: usize,

    /// WS の張り替え確率 β．
    #[arg(long, default_value_t = 0.1)]
    ws_beta: f64,

    /// ER の辺確率 p．
    #[arg(long, default_value_t = 0.02)]
    er_p: f64,

    /// ABM 同化率 α．
    #[arg(long, default_value_t = 0.3)]
    alpha: f64,

    /// ABM 信頼境界 ε．
    #[arg(long, default_value_t = 0.4)]
    epsilon: f64,

    /// 動員判定の態度しきい値．
    #[arg(long, default_value_t = 0.5)]
    mobilization_threshold: f64,

    /// 1 実行あたりの最大 LLM 呼び出し数．
    #[arg(long, default_value_t = 5000)]
    llm_budget: usize,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0; 再現性のため)．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// コア post の stance → 態度 写像 (deterministic = 既定の決定論的分類器 /
    /// llm = 外部 LLM stance 注釈; live バックエンドが必要)．
    #[arg(long, default_value = "deterministic")]
    stance_annotator: String,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// データセット (metoo / roe / blm)．
    #[arg(long, default_value = "metoo")]
    dataset: String,

    /// コア比率スイープ下限．
    #[arg(long, default_value_t = 0.0)]
    core_ratio_min: f64,

    /// コア比率スイープ上限．
    #[arg(long, default_value_t = 0.5)]
    core_ratio_max: f64,

    /// コア比率スイープ刻み．
    #[arg(long, default_value_t = 0.1)]
    core_ratio_step: f64,

    /// カンマ区切りの ABM 種別リスト．
    #[arg(long, default_value = "bc,hk,sj,lorenz")]
    abm_values: String,

    /// カンマ区切りのネットワーク構造リスト．
    #[arg(long, default_value = "ba")]
    network_values: String,

    /// エージェント数 N．
    #[arg(long, default_value_t = 1000)]
    n_agents: usize,

    /// タイムステップ数 T．
    #[arg(long, default_value_t = 14)]
    steps: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 10)]
    runs: usize,

    /// 1 実行あたりの最大 LLM 呼び出し数．
    #[arg(long, default_value_t = 5000)]
    llm_budget: usize,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// 照合する運動データセット (カンマ区切り; metoo / roe / blm)．
    #[arg(long, default_value = "metoo,roe,blm")]
    datasets: String,

    /// 比較する周辺 ABM 種別 (カンマ区切り; bc / hk / sj / lorenz)．
    #[arg(long, default_value = "bc,hk,sj,lorenz")]
    abm_values: String,

    /// ハイブリッド条件のコア比率 (純 ABM 条件は常に 0.0)．
    #[arg(long, default_value_t = 0.3)]
    core_ratio: f64,

    /// エージェント数 N．
    #[arg(long, default_value_t = 600)]
    n_agents: usize,

    /// タイムステップ数 T．
    #[arg(long, default_value_t = 14)]
    steps: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// ネットワーク構造 (ba / ws / er)．
    #[arg(long, default_value = "ba")]
    network: String,

    /// 乱数シード基点 (各条件・試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// ライブ LLM を呼ばず決定論的 scripted mock で駆動する (オフライン検証用)．
    /// サンドボックス・CI では `--mock` を付ける (純 ABM 条件は mock 不要)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// LLM 生成温度 (live 時のみ)．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード (live 時のみ)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// コア post の stance → 態度 写像 (deterministic / llm)．
    #[arg(long, default_value = "deterministic")]
    stance_annotator: String,

    /// プロンプト→応答キャッシュの保存先 (live 時のみ; 全条件で共有)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 軽量モード (N・runs・steps を縮小; 動作確認用)．
    #[arg(long, default_value_t = false)]
    quick: bool,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// コア層の決定を数える stage を，メカニズムの中から突ける形にして渡す．
///
/// 数える場所は [`DecisionMechanism`](hisim_simulation::mechanisms::DecisionMechanism)
/// の中である．メカニズムはエンジンへ `Box<dyn Mechanism<_>>` として入るので
/// `'static` であり，呼び出し側の `Stage` を借用できない — そこで `Rc` で共有し，
/// 走り終えたあとに [`close_shared`] で取り出して閉じる．
fn share_stage(stage: Stage) -> (Rc<RefCell<Option<Stage>>>, DecisionObserver) {
    let cell = Rc::new(RefCell::new(Some(stage)));
    let observer: DecisionObserver = {
        let cell = Rc::clone(&cell);
        Rc::new(RefCell::new(move || {
            if let Some(stage) = cell.borrow_mut().as_mut() {
                stage.tick();
            }
        }))
    };
    (cell, observer)
}

/// 共有していた stage を取り出して閉じる．
///
/// manifest.csv は `finish()` で封をされる．その後に 1 行足せば，manifest が
/// 食い違うダイジェストを持つことになる．
fn close_shared(cell: &Rc<RefCell<Option<Stage>>>) {
    if let Some(stage) = cell.borrow_mut().take() {
        stage.close();
    }
}

/// スイープ親 run の実験条件 (グリッド定義そのもの)．
#[derive(Serialize)]
struct SweepParameters {
    dataset: String,
    core_ratio_values: Vec<f64>,
    abm_values: Vec<String>,
    network_values: Vec<String>,
    n_agents: usize,
    steps: usize,
    runs: usize,
    llm_budget: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// スイープの子 run (network × abm × core-ratio の 1 点) の実験条件．
///
/// `run` の条件に `runs` が付いた形で，`run` とは別のサブコマンド名を持つ．
/// 同じ `run` を名乗らせると，「1 本のシミュレーション」と「同一条件の
/// `runs` 本」という中身の違う 2 つが 1 つの名前に同居し，`runvault path
/// --subcommand run` がどちらを返すか分からなくなる．
#[derive(Serialize)]
struct SweepPointParameters {
    dataset: String,
    network: String,
    abm: String,
    core_ratio: f64,
    n_agents: usize,
    steps: usize,
    runs: usize,
    llm_budget: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// `reproduce` run の実験条件．
///
/// `n_agents` / `runs` / `steps` は `--quick` を反映した **実際に回した値**で，
/// `--quick` そのものは持たない (同じ条件なら同じ config_hash になる)．
#[derive(Serialize)]
struct ReproduceParameters {
    datasets: Vec<String>,
    abm_values: Vec<String>,
    core_ratio: f64,
    n_agents: usize,
    steps: usize,
    runs: usize,
    network: String,
    stance: String,
    mock: bool,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// コア比率スイープの値列を [min, max] を step 刻みで生成する．
fn ratio_values(min: f64, max: f64, step: f64) -> Vec<f64> {
    let mut out = Vec::new();
    if step <= 0.0 {
        out.push(min);
        return out;
    }
    let mut v = min;
    while v <= max + 1e-9 {
        out.push((v * 1000.0).round() / 1000.0);
        v += step;
    }
    out
}

/// CLI 引数から `NetworkConfig` を組み立てる (種別を差し替えられるよう分離)．
fn network_config(
    kind: NetworkKind,
    ba_m: usize,
    ws_k: usize,
    ws_beta: f64,
    er_p: f64,
) -> NetworkConfig {
    NetworkConfig {
        kind,
        ba_m,
        ws_k,
        ws_beta,
        er_p,
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let abm_model = parse_abm(&args.abm).unwrap_or_else(|e| panic!("{}", e));
    let net_kind = parse_network(&args.network).unwrap_or_else(|e| panic!("{}", e));
    let stance = parse_stance_mode(&args.stance_annotator).unwrap_or_else(|e| panic!("{}", e));

    // シードを実体化してから記録する．--seed 省略時にシミュレーション側で
    // rand::random に落とすと，実際に使われたシードがどこにも残らない．
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);

    let cfg = Config {
        dataset: args.dataset.clone(),
        n_agents: args.n_agents,
        core_ratio: args.core_ratio,
        steps: args.steps,
        network: network_config(net_kind, args.ba_m, args.ws_k, args.ws_beta, args.er_p),
        abm: AbmParams {
            model: abm_model,
            alpha: args.alpha,
            epsilon: args.epsilon,
            ..AbmParams::default()
        },
        mobilization_threshold: args.mobilization_threshold,
        llm_budget: args.llm_budget,
        seed: Some(seed),
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        stance,
    };

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    // LLM クライアントは run を開始する前に組む．`llm` ブロックに書くモデル名と
    // endpoint は，実際に応答するバックエンドから採らないと意味を持たない．
    let client =
        build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"));
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let parameters = cfg.to_run_config_json(seed);
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(seed)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    println!("=== Mou et al. (2024) HiSim 大規模社会運動シミュレーション 再現実験 ===");
    println!(
        "dataset: {} | abm: {} | core-ratio: {} | N: {} | T: {} | network: {}",
        cfg.dataset,
        cfg.abm.model.label(),
        cfg.core_ratio,
        cfg.n_agents,
        cfg.steps,
        cfg.network.kind.label(),
    );
    println!(
        "seed: {} | llm-budget: {} | stance: {} | LLM: temp={} llm_seed={} cache={}",
        seed,
        cfg.llm_budget,
        cfg.stance.label(),
        cfg.llm.temperature,
        cfg.llm.seed,
        args.cache_path
    );
    println!("出力先: {}", rv.dir().display());
    println!("-----------------------------------------------------------------");

    // 進捗の単位は設定が決める．コア層があるならその 1 体の決定が費用で，無ければ
    // 決定は一度も起きないので 1 タイムステップが費用になる．
    //
    // コア層あり (既定 --core-ratio 0.3 --n-agents 1000): 1 ステップが 300 回の
    // LLM 呼び出しで，ローカル Ollama (llama3.2) の実測 1.36s/回 では 1 ステップ
    // 約 6 分 48 秒，14 ステップで約 1 時間 35 分になる．ステップを数えたら
    // 7 分近く同じ数字が出続ける．
    //
    // 純 ABM (--core-ratio 0.0): LLM 呼び出しは 0 回で，N=1000 T=14 の実測は 0s．
    // それでも --n-agents は伸ばせるので，ステップを数える．
    //
    // どちらも分母を持たない．[`AggregateMechanism`] が bias の変化が tol 未満に
    // なった時点で `request_stop` するので，`steps` は «到達しない上限» である．
    let core_driven = cfg.core_ratio > 0.0;
    let decisions = if core_driven {
        let (cell, observer) = share_stage(rv.unbounded_stage("decisions"));
        (Some(cell), observer)
    } else {
        (None, no_observer())
    };
    let mut step_stage = if core_driven {
        None
    } else {
        Some(rv.unbounded_stage("steps"))
    };

    let result = run_with_client_observed(&cfg, client, decisions.1, |_| {
        if let Some(stage) = step_stage.as_mut() {
            stage.tick();
        }
    })
    .unwrap_or_else(|e| panic!("実行に失敗: {}", e));

    // stage は rv.finish() より先に閉じる (manifest.csv は finish() で封をされる)．
    if let Some(cell) = &decisions.0 {
        close_shared(cell);
    }
    if let Some(stage) = step_stage {
        stage.close();
    }

    record::log_simulation(&mut rv, &result);

    let last = result.metrics_history.last().unwrap();
    println!(
        "収束: {} | step: {}",
        if result.converged { "Yes" } else { "No" },
        result.final_step
    );
    println!(
        "最終 macro_bias: {:.4} | diversity: {:.4} | mobilized: {} | polarization: {:.4} | core_influence: {:.4}",
        last.macro_bias, last.macro_diversity, last.mobilized, last.polarization, last.core_influence,
    );
    println!(
        "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("メトリクス → {}/metrics.csv", dir.display());
    println!("設定       → {}/config.json", dir.display());
    println!("LLM メタ   → {}/run.json (llm ブロック)", dir.display());
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let core_ratio_values = ratio_values(
        args.core_ratio_min,
        args.core_ratio_max,
        args.core_ratio_step,
    );
    let abm_models: Vec<AbmModel> = split_csv(&args.abm_values)
        .iter()
        .map(|s| parse_abm(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let net_kinds: Vec<NetworkKind> = split_csv(&args.network_values)
        .iter()
        .map(|s| parse_network(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = core_ratio_values.len() * abm_models.len() * net_kinds.len() * args.runs;

    let llm_settings = LlmSettings {
        temperature: args.llm_temperature,
        seed: args.llm_seed,
        cache_path: Some(args.cache_path.clone()),
    };
    // 全条件が同じバックエンドを使うので，`llm` ブロックは 1 度組んで子 run へ配る．
    // 名乗る名前を知っているのはクライアントだけなので，回す前に 1 つ組んで訊く．
    let llm = {
        let probe = build_live_client(&llm_settings)
            .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"));
        record::llm_block(
            probe.inner().model(),
            probe.inner().endpoint(),
            llm_settings.temperature,
        )
    };

    let sweep_parameters = SweepParameters {
        dataset: args.dataset.clone(),
        core_ratio_values: core_ratio_values.clone(),
        abm_values: abm_models.iter().map(|m| m.label().to_string()).collect(),
        network_values: net_kinds.iter().map(|n| n.label().to_string()).collect(),
        n_agents: args.n_agents,
        steps: args.steps,
        runs: args.runs,
        llm_budget: args.llm_budget,
        seed: args.seed,
        llm_temperature: args.llm_temperature,
        llm_seed: args.llm_seed,
    };

    // 親 run: グリッド定義そのものを parameters に持つ．個別条件の指標は書かない．
    // 親は 1 本のシミュレーションではないので master_seed を名乗らず，基点シードは
    // /parameters.seed と seed_pointers 経由で execution_hash に残る．
    // sweep_id は runvault が親の run_slug で埋める．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .llm(llm.clone())
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Mou et al. (2024) HiSim パラメータスイープ (core-ratio × abm × network) ===");
    println!(
        "dataset: {} | core-ratio: {} 種 | abm: {} 種 | network: {} 種 | 試行: {} | 合計: {} 実行",
        args.dataset,
        core_ratio_values.len(),
        abm_models.len(),
        net_kinds.len(),
        args.runs,
        n_total,
    );
    println!("シード (base): {}", args.seed);
    println!("出力先: {}", parent.dir().display());
    println!("-----------------------------------------------------------------");

    // ABM 種別ごとの平均分極化 (最後に出す要約)．試行ごとの値は子 run の
    // events.jsonl が正本なので，ここでは表示のためだけに積む．
    let mut polarization_by_abm: Vec<(AbmModel, Vec<f64>)> =
        abm_models.iter().map(|&m| (m, Vec::new())).collect();
    let mut done = 0usize;

    // 1 本のスイープに，費用の桁が 5 つ違う 2 種類の仕事が同居する．core-ratio > 0
    // のセルは 1 試行が数千回の LLM 呼び出しで，既定のグリッド (core-ratio
    // 0.0..0.5 step 0.1 × abm 4 種 × runs 10) を live で回すと 736,000 回 =
    // 実測 1.36s/回で約 11.6 日になる．core-ratio = 0 のセルは LLM を 1 度も呼ばず，
    // 純 ABM のスイープ 40 試行が実測 2s で終わる．
    //
    // だから 1 つの stage にまとめず，違うものを名前で分ける．重み付けはしない —
    // 1 試行の重みは走らせるまで分からず，速い側が遅い側の残り時間を決めてしまう．
    //
    // - `decisions`: コア層 1 体の決定 (core-ratio > 0 のセルの費用)．何回鳴るかは
    //   予算と収束停止で決まるので分母を持てない．
    // - `abm-trials`: 純 ABM セルの試行 1 本．試行の «本数» は途中で変わらないので
    //   分母は正確である (各試行が収束で早く止まっても本数は減らない)．
    //
    // 分母は値列の `len()` から採る．core_ratio_step は 0.1 のような二進で表せない
    // 刻みで，範囲を割って本数を出すと 1 本ずれる．
    let abm_ratio_count = core_ratio_values.iter().filter(|&&r| r <= 0.0).count();
    let abm_trials_total = abm_ratio_count * abm_models.len() * net_kinds.len() * args.runs;
    let mut abm_trials = if abm_trials_total > 0 {
        Some(parent.stage("abm-trials", abm_trials_total))
    } else {
        None
    };
    let decisions = if core_ratio_values.iter().any(|&r| r > 0.0) {
        let (cell, observer) = share_stage(parent.unbounded_stage("decisions"));
        (Some(cell), observer)
    } else {
        (None, no_observer())
    };

    for &net_kind in &net_kinds {
        for &abm_model in &abm_models {
            for &core_ratio in &core_ratio_values {
                let params = SweepPointParameters {
                    dataset: args.dataset.clone(),
                    network: net_kind.label().to_string(),
                    abm: abm_model.label().to_string(),
                    core_ratio,
                    n_agents: args.n_agents,
                    steps: args.steps,
                    runs: args.runs,
                    llm_budget: args.llm_budget,
                    seed: args.seed,
                    llm_temperature: args.llm_temperature,
                    llm_seed: args.llm_seed,
                };

                // 子は «その条件の試行群» そのもの．master_seed は親と同じ基点で，
                // 条件が違えば config_hash が違うので run としては別物になる．
                // 同じ条件の繰り返しは無いので replicate_index は 0．
                let mut child = Run::start(
                    RunOptions::new(EXPERIMENT, "sweep-point")
                        .repo_id(REPO_ID)
                        .domain(DOMAIN)
                        .results_root(&args.output_dir)
                        .parameters(&params)
                        .expect("runvault: 子 run の parameters の組み立てに失敗")
                        .seed_pointers(["/seed"])
                        .master_seed(args.seed)
                        .replicate_index(0)
                        .llm(llm.clone())
                        .lineage(Lineage {
                            sweep_id: Some(sweep_id.clone()),
                            parent_run_uid: Some(parent_run_uid.clone()),
                            ..Default::default()
                        })
                        .replication(record::replication()),
                )
                .expect("runvault: 子 run の開始に失敗");

                let mut trials: Vec<record::TrialOutcome> = Vec::with_capacity(args.runs);
                for run_idx in 0..args.runs {
                    // 各 (network, abm, core_ratio, run) に独立なシードを派生させる．
                    let seed = record::sweep_trial_seed(
                        args.seed,
                        net_kind.label(),
                        abm_model.label(),
                        core_ratio,
                        run_idx,
                    );

                    let cfg = Config {
                        dataset: args.dataset.clone(),
                        n_agents: args.n_agents,
                        core_ratio,
                        steps: args.steps,
                        network: NetworkConfig {
                            kind: net_kind,
                            ..NetworkConfig::default()
                        },
                        abm: AbmParams {
                            model: abm_model,
                            ..AbmParams::default()
                        },
                        mobilization_threshold: 0.5,
                        llm_budget: args.llm_budget,
                        seed: Some(seed),
                        llm: llm_settings.clone(),
                        stance: StanceMode::default(),
                    };

                    let client = build_live_client(&cfg.llm)
                        .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"));
                    let result =
                        run_with_client_observed(&cfg, client, Rc::clone(&decisions.1), |_| {})
                            .unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                    // 純 ABM のセルだけがここを数える (`abm_trials_total` と同じ条件)．
                    if core_ratio <= 0.0 {
                        if let Some(stage) = abm_trials.as_mut() {
                            stage.tick();
                        }
                    }

                    // 旧 sweep_summary.csv の 1 行が terminal 行 1 本に対応する．
                    // metrics.csv に入れると (run_uid, step, scope, name) が重複する．
                    record::log_trial(
                        &mut child,
                        &format!("trial-{run_idx}"),
                        seed,
                        args.steps,
                        &result,
                    );
                    let outcome = record::TrialOutcome::from_result(&result);
                    if let Some((_, values)) = polarization_by_abm
                        .iter_mut()
                        .find(|(m, _)| *m == abm_model)
                    {
                        values.push(outcome.polarization);
                    }
                    trials.push(outcome);

                    done += 1;
                }
                record::log_condition_summary(&mut child, &trials);
                child.finish().expect("runvault: 子 run の完了に失敗");

                println!(
                    "[{}/{}] network={} abm={} core-ratio={:.2} 完了 ({} 試行)",
                    done,
                    n_total,
                    net_kind.label(),
                    abm_model.label(),
                    core_ratio,
                    args.runs,
                );
            }
        }
    }

    // stage は finish() より先に閉じる (manifest.csv は finish() で封をされる)．
    if let Some(stage) = abm_trials {
        stage.close();
    }
    if let Some(cell) = &decisions.0 {
        close_shared(cell);
    }

    let parent_dir = parent.finish().expect("runvault: sweep 親 run の完了に失敗");

    println!("=================================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("ABM 種別別の平均 分極化 polarization:");
    for (abm_model, values) in &polarization_by_abm {
        if values.is_empty() {
            continue;
        }
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        println!("  abm={:<7} → polarization̄ = {:.4}", abm_model.label(), avg);
    }
    println!("-----------------------------------------------------------------");
    println!("親 run  → {}", parent_dir.display());
    println!("試行の値 → 各子 run の events.jsonl (terminal 行)");
}

// ---------------------------------------------------------------------------
// reproduce — Table 2/3 + SoMoSiMu-Bench 照合
// ---------------------------------------------------------------------------

/// 1 レジーム (hybrid / pure-abm) × ABM を `runs` 回回した集計セル (Table 3 の元)．
///
/// 試行ごとの値ではなく試行平均だけを持つ (旧 `reproduce_summary.json` と同じ粒度)．
#[derive(Clone)]
struct ReproCell {
    /// 条件ラベル (例: "hybrid_bc" / "pureabm_lorenz")．指標名の接頭辞にもなる．
    label: String,
    /// レジーム ("hybrid" = LLM コア + ABM 周辺 / "pure-abm" = core-ratio 0)．
    regime: String,
    /// 周辺 ABM 種別．
    abm: String,
    /// 試行平均の最終 macro_bias (集団態度の偏り; Table 3 Bias)．
    mean_final_bias: f64,
    /// 試行平均の最終 macro_diversity (意見多様性; Table 3 Div.)．
    mean_final_diversity: f64,
    /// 試行平均の最終 polarization (二極化)．
    mean_final_polarization: f64,
    /// 試行平均の最終 正規化動員 (mobilized / N)．
    mean_final_mobilization: f64,
    /// 試行平均の «動員の伸び» (最終 − 初期 mobilized / N; 正なら運動が拡大)．
    mean_mobilization_gain: f64,
    /// 試行平均の総 LLM 呼び出し数 (pure-abm は 0)．
    mean_llm_calls: f64,
}

impl ReproCell {
    /// run スコープ指標として書く値の並び (接頭辞は [`ReproCell::label`])．
    fn metrics(&self) -> [(&'static str, f64); 6] {
        [
            ("mean_final_bias", self.mean_final_bias),
            ("mean_final_diversity", self.mean_final_diversity),
            ("mean_final_polarization", self.mean_final_polarization),
            ("mean_final_mobilization", self.mean_final_mobilization),
            ("mean_mobilization_gain", self.mean_mobilization_gain),
            ("mean_llm_calls", self.mean_llm_calls),
        ]
    }
}

/// 観測値と論文の定性的知見を突き合わせた 1 アンカー (`events.jsonl` へ書く)．
///
/// `name` は runvault の指標名にもなるので slug (小文字・数字・`_`・`-`・`.`) に
/// 収める．比較の向きは名前の中に `_ge_` のように織り込み，帯そのものは
/// `target_lo` / `target_hi` が持つ．
#[derive(Serialize)]
struct ReproAnchor {
    name: String,
    paper: String,
    observed: f64,
    target_lo: f64,
    /// 帯の上限．上限なしは `None`．
    ///
    /// `f64::INFINITY` は JSON で表現できず `null` に潰れるので，«上限が無い» と
    /// «値を書き忘れた» が区別できなくなる．最初から `Option` で持つ．
    target_hi: Option<f64>,
    pass: bool,
}

/// SoMoSiMu-Bench の整合判定 1 行 (`events.jsonl` へ書く)．
///
/// [`hisim_simulation::bench::AlignmentRow`] に，どの運動のどの参照との比較かを
/// 添えたもの．参照値はこの再現実装が置いた合成アンカーであって論文の報告値では
/// ないので，出典を要求する `reference.csv` ではなくここに置く．
#[derive(Serialize)]
struct BenchAlignmentEvent<'a> {
    movement: &'a str,
    reference_source: &'a str,
    metric: &'a str,
    observed: f64,
    reference: f64,
    abs_error: f64,
    tolerance: f64,
    aligned: bool,
}

/// 1 セルの実行結果 (集計セル・bench 用の運動指標・代表 run の履歴)．
struct ReproCellResult {
    cell: ReproCell,
    /// 試行ごとの運動指標 (bench 照合の観測系列の材料)．
    movement_runs: Vec<MovementMetrics>,
    /// 代表 run (run 0) のステップごとの履歴．
    representative: Vec<StepMetrics>,
}

/// 1 レジーム × ABM を `runs` 回実行して集計セルを作る．
#[allow(clippy::too_many_arguments)]
fn run_repro_cell(
    label: &str,
    regime: &str,
    abm_model: AbmModel,
    core_ratio: f64,
    net_kind: NetworkKind,
    dataset: &str,
    n_agents: usize,
    steps: usize,
    stance: StanceMode,
    runs: usize,
    root_seed: u64,
    mock: bool,
    llm: &LlmSettings,
    observer: DecisionObserver,
    on_abm_trial: &mut dyn FnMut(),
) -> ReproCellResult {
    let mut final_bias = 0.0;
    let mut final_div = 0.0;
    let mut final_pol = 0.0;
    let mut final_mob = 0.0;
    let mut mob_gain = 0.0;
    let mut llm_calls = 0.0;
    let mut movement_runs: Vec<MovementMetrics> = Vec::with_capacity(runs);
    let mut representative: Vec<StepMetrics> = Vec::new();

    for run_idx in 0..runs {
        let seed =
            record::repro_trial_seed(root_seed, regime, abm_model.label(), dataset, run_idx);
        let cfg = Config {
            dataset: dataset.to_string(),
            n_agents,
            core_ratio,
            steps,
            network: NetworkConfig {
                kind: net_kind,
                ..NetworkConfig::default()
            },
            abm: AbmParams {
                model: abm_model,
                ..AbmParams::default()
            },
            mobilization_threshold: 0.5,
            llm_budget: 1_000_000,
            seed: Some(seed),
            llm: llm.clone(),
            stance,
        };

        // pure-abm (core_ratio 0) は LLM を一切呼ばないので mock も live も不要．
        let client: HiSimClient = if core_ratio == 0.0 || mock {
            build_reproduce_client()
        } else {
            build_live_client(&cfg.llm)
                .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗 ({label}): {e}"))
        };
        let result: SimulationResult =
            run_with_client_observed(&cfg, client, Rc::clone(&observer), |_| {})
                .unwrap_or_else(|e| panic!("実行に失敗 ({label}): {e}"));
        // 純 ABM のセルだけが試行を数える (LLM を 1 度も呼ばないので `observer` は
        // 鳴らない)．`cmd_reproduce` の `abm_trials_total` と同じ条件である．
        if core_ratio <= 0.0 {
            on_abm_trial();
        }

        let first = result.metrics_history.first().unwrap();
        let last = result.metrics_history.last().unwrap();
        let nf = n_agents as f64;
        final_bias += last.macro_bias;
        final_div += last.macro_diversity;
        final_pol += last.polarization;
        final_mob += last.mobilized as f64 / nf;
        mob_gain += (last.mobilized as f64 - first.mobilized as f64) / nf;
        llm_calls += result.metadata.total() as f64;
        movement_runs.push(MovementMetrics::from_history(
            &result.metrics_history,
            n_agents,
        ));
        if run_idx == 0 {
            representative = result.metrics_history.clone();
        }
    }

    let n = runs.max(1) as f64;
    let cell = ReproCell {
        label: label.to_string(),
        regime: regime.to_string(),
        abm: abm_model.label().to_string(),
        mean_final_bias: final_bias / n,
        mean_final_diversity: final_div / n,
        mean_final_polarization: final_pol / n,
        mean_final_mobilization: final_mob / n,
        mean_mobilization_gain: mob_gain / n,
        mean_llm_calls: llm_calls / n,
    };
    ReproCellResult {
        cell,
        movement_runs,
        representative,
    }
}

/// 複数 run の運動指標を平均する (bench 照合の観測値)．
fn mean_movement(runs: &[MovementMetrics]) -> MovementMetrics {
    let n = runs.len().max(1) as f64;
    let mut acc = MovementMetrics {
        mobilization_peak: 0.0,
        peak_step: 0,
        final_mobilization: 0.0,
        final_bias: 0.0,
        final_polarization: 0.0,
        sustain_ratio: 0.0,
    };
    let mut peak_step_sum = 0.0;
    for m in runs {
        acc.mobilization_peak += m.mobilization_peak;
        peak_step_sum += m.peak_step as f64;
        acc.final_mobilization += m.final_mobilization;
        acc.final_bias += m.final_bias;
        acc.final_polarization += m.final_polarization;
        acc.sustain_ratio += m.sustain_ratio;
    }
    acc.mobilization_peak /= n;
    acc.peak_step = (peak_step_sum / n).round() as usize;
    acc.final_mobilization /= n;
    acc.final_bias /= n;
    acc.final_polarization /= n;
    acc.sustain_ratio /= n;
    acc
}

/// 運動指標を run スコープ指標の並びへ落とす．
fn movement_metrics(m: &MovementMetrics) -> [(&'static str, f64); 6] {
    [
        ("mobilization_peak", m.mobilization_peak),
        ("peak_step", m.peak_step as f64),
        ("final_mobilization", m.final_mobilization),
        ("final_bias", m.final_bias),
        ("final_polarization", m.final_polarization),
        ("sustain_ratio", m.sustain_ratio),
    ]
}

fn cmd_reproduce(args: ReproduceArgs) {
    let datasets = split_csv(&args.datasets);
    let abm_models: Vec<AbmModel> = split_csv(&args.abm_values)
        .iter()
        .map(|s| parse_abm(s).unwrap_or_else(|e| panic!("{e}")))
        .collect();
    let net_kind = parse_network(&args.network).unwrap_or_else(|e| panic!("{e}"));
    let stance = parse_stance_mode(&args.stance_annotator).unwrap_or_else(|e| panic!("{e}"));

    // quick モードは軽量化 (動作確認用)．
    let n_agents = if args.quick { 80 } else { args.n_agents };
    let runs = if args.quick { 2 } else { args.runs };
    let steps = if args.quick { 8 } else { args.steps };

    if !args.mock {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let llm = LlmSettings {
        temperature: args.llm_temperature,
        seed: args.llm_seed,
        cache_path: if args.mock {
            None
        } else {
            Some(args.cache_path.clone())
        },
    };

    // 名乗る名前を知っているのはクライアントだけなので，回す前に 1 つ組んで訊く．
    // `--mock` なら scripted mock が，live ならフォールバッククライアントが答える
    // (live でも pure-abm セルは LLM を 1 度も呼ばないが，ハイブリッドセルが呼ぶ
    // のはこのバックエンドである)．
    let llm_block = {
        let probe: HiSimClient = if args.mock {
            build_reproduce_client()
        } else {
            build_live_client(&llm)
                .unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"))
        };
        record::llm_block(probe.inner().model(), probe.inner().endpoint(), llm.temperature)
    };

    let parameters = ReproduceParameters {
        datasets: datasets.clone(),
        abm_values: abm_models.iter().map(|m| m.label().to_string()).collect(),
        core_ratio: args.core_ratio,
        n_agents,
        steps,
        runs,
        network: net_kind.label().to_string(),
        stance: stance.label().to_string(),
        mock: args.mock,
        seed: args.seed,
        llm_temperature: args.llm_temperature,
        llm_seed: args.llm_seed,
    };

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(args.seed)
            .llm(llm_block)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    println!("=== Mou et al. (2024) HiSim — Table 2/3 + SoMoSiMu-Bench 一括再現 ===");
    println!(
        "datasets: {} | abm: {} 種 | N: {} | T: {} | runs: {} | network: {} | stance: {} | mode: {}",
        datasets.join(","),
        abm_models.len(),
        n_agents,
        steps,
        runs,
        net_kind.label(),
        stance.label(),
        if args.mock { "MOCK" } else { "LIVE" },
    );
    println!("出力先: {}", rv.dir().display());
    println!("-----------------------------------------------------------------");

    // 進捗の単位を 2 つに分ける．費用の桁が違うものを 1 つの stage にまとめると，
    // 速い側が遅い側の残り時間を決めてしまう．重み付けはしない — 1 試行の重みは
    // 走らせるまで分からない．
    //
    // - `decisions` (分母なし): hybrid セルのコア層 1 体の決定．既定 (N=600
    //   core-ratio 0.3 runs 5 abm 4 種) の live では 4×5×180×14 = 50,400 回の LLM
    //   呼び出しで，実測 1.36s/回では約 19 時間になる．1 ステップだけでも 180 回 =
    //   約 4 分なので，ステップやセルでは粗すぎる．
    // - `abm-trials` (分母あり): 純 ABM セルの試行 1 本．LLM を 1 度も呼ばず，
    //   `reproduce --mock` 全体が実測 2s で終わる速さである．試行の «本数» は
    //   セル数 × runs で確定していて，各試行が収束で早く止まっても減らない．
    //
    // --core-ratio 0.0 を渡すと hybrid セルも LLM を呼ばなくなるので，そのぶんを
    // 純 ABM 側の本数に加える (`run_repro_cell` が `core_ratio <= 0.0` で数える条件と
    // 一致させる)．
    let hybrid_is_abm = args.core_ratio <= 0.0;
    let abm_cells =
        abm_models.len() + datasets.len() + if hybrid_is_abm { abm_models.len() } else { 0 };
    let mut abm_trials = if abm_cells * runs > 0 {
        Some(rv.stage("abm-trials", abm_cells * runs))
    } else {
        None
    };
    let decisions = if hybrid_is_abm {
        (None, no_observer())
    } else {
        let (cell, observer) = share_stage(rv.unbounded_stage("decisions"));
        (Some(cell), observer)
    };
    // --- Table 3: hybrid vs pure-abm の ABM 別行列 (代表 dataset = 先頭) ---
    // 純 ABM 経路 (core-ratio 0) は LLM 0 呼び出しで完全決定論的 = オフライン検証経路．
    let table3_dataset = datasets
        .first()
        .cloned()
        .unwrap_or_else(|| "metoo".to_string());
    let mut table3_cells: Vec<ReproCell> = Vec::new();
    for &abm_model in &abm_models {
        for &(regime, ratio) in &[("pure-abm", 0.0), ("hybrid", args.core_ratio)] {
            let label = format!("{}_{}", regime.replace('-', ""), abm_model.label());
            let out = run_repro_cell(
                &label,
                regime,
                abm_model,
                ratio,
                net_kind,
                &table3_dataset,
                n_agents,
                steps,
                stance,
                runs,
                args.seed,
                args.mock,
                &llm,
                Rc::clone(&decisions.1),
                &mut || {
                    if let Some(stage) = abm_trials.as_mut() {
                        stage.tick();
                    }
                },
            );
            // 11 条件が 1 本の run に同居するので，(step, scope, name) が衝突しない
            // よう条件ラベルを名前に付ける．
            record::log_history(&mut rv, Some(&label), &out.representative);
            record::log_prefixed(&mut rv, &label, &out.cell.metrics());
            table3_cells.push(out.cell);
        }
    }

    // --- Table 2: SoMoSiMu-Bench 照合 (dataset 別; pure-abm BC を観測系列とする) ---
    // 純 ABM の動員ダイナミクスを各運動の合成参照と照合する (オフライン経路)．
    let bench_abm = AbmModel::Bc;
    let mut bench_comparisons: Vec<hisim_simulation::bench::BenchComparison> = Vec::new();
    for ds in &datasets {
        let label = format!("bench_{ds}");
        let out = run_repro_cell(
            &label,
            "pure-abm",
            bench_abm,
            0.0,
            net_kind,
            ds,
            n_agents,
            steps,
            stance,
            runs,
            args.seed,
            args.mock,
            &llm,
            Rc::clone(&decisions.1),
            &mut || {
                if let Some(stage) = abm_trials.as_mut() {
                    stage.tick();
                }
            },
        );
        record::log_history(&mut rv, Some(&label), &out.representative);

        let observed = mean_movement(&out.movement_runs);
        record::log_prefixed(&mut rv, &label, &movement_metrics(&observed));

        let reference = reference_curve(ds, steps);
        let comparison = compare_to_bench(ds, observed, &reference);
        record::log_prefixed(
            &mut rv,
            &label,
            &[
                ("n_aligned", comparison.n_aligned as f64),
                ("n_total", comparison.n_total as f64),
            ],
        );
        for row in &comparison.rows {
            record::log_verdict(
                &mut rv,
                BENCH_EVENT,
                &row.metric,
                &BenchAlignmentEvent {
                    movement: &comparison.movement,
                    reference_source: &comparison.reference_source,
                    metric: &row.metric,
                    observed: row.observed,
                    reference: row.reference,
                    abs_error: row.abs_error,
                    tolerance: row.tolerance,
                    aligned: row.aligned,
                },
            );
        }
        bench_comparisons.push(comparison);
    }

    // --- アンカー評価 (論文 Table 2/3 の定性的知見) ---
    let cell = |regime: &str, abm: &str| -> &ReproCell {
        table3_cells
            .iter()
            .find(|c| c.regime == regime && c.abm == abm)
            .unwrap_or_else(|| panic!("セル {regime}/{abm} が見つかりません"))
    };
    let bc_pure = cell("pure-abm", "bc");
    let bc_hybrid = cell("hybrid", "bc");
    let lorenz_pure = cell("pure-abm", "lorenz");
    let sj_pure = cell("pure-abm", "sj");
    let hk_pure = cell("pure-abm", "hk");

    let mut anchors: Vec<ReproAnchor> = Vec::new();
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: Option<f64>| {
        anchors.push(ReproAnchor {
            name: name.to_string(),
            paper: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && hi.is_none_or(|h| obs <= h),
        });
    };

    // T3-A: pure-abm は LLM を一切呼ばない (オフライン検証可能経路)．
    push(
        "pure_abm_zero_llm_calls",
        "core-ratio 0 = no LLM",
        bc_pure.mean_llm_calls,
        0.0,
        Some(0.0),
    );
    // T3-B: 二極化の順序 BC ≤ {SJ, Lorenz} (論文 §A: BC/HK は合意，SJ/Lorenz は分極)．
    push(
        "polarization_lorenz_ge_bc",
        "Lorenz polarizes vs BC consensus",
        lorenz_pure.mean_final_polarization - bc_pure.mean_final_polarization,
        -1e-9,
        None,
    );
    push(
        "polarization_sj_ge_bc",
        "SJ polarizes vs BC consensus",
        sj_pure.mean_final_polarization - bc_pure.mean_final_polarization,
        -1e-9,
        None,
    );
    // T3-C: BC/HK は合意寄り (低分極; 分極 < 0.5)．
    push(
        "bc_low_polarization",
        "BC reaches consensus (low polarization)",
        bc_pure.mean_final_polarization,
        0.0,
        Some(0.5),
    );
    push(
        "hk_low_polarization",
        "HK reaches consensus (low polarization)",
        hk_pure.mean_final_polarization,
        0.0,
        Some(0.5),
    );
    // T3-D: ハイブリッド (LLM コア) は純 ABM より動員を牽引する
    //   (mock では支持コアが call-to-action を発信 → 動員の伸びが純 ABM 以上)．
    push(
        "hybrid_amplifies_mobilization",
        "core LLM drives mobilization (gain_hybrid - gain_pureabm >= 0)",
        bc_hybrid.mean_mobilization_gain - bc_pure.mean_mobilization_gain,
        -1e-9,
        None,
    );

    // 観測量そのものは run 全体を 1 つの値で表す数なので指標に書く．判定 (PASS/off)
    // と帯はカテゴリ・自前のアンカーなので events.jsonl へ．
    let observed: Vec<(&str, f64)> = anchors
        .iter()
        .map(|a| (a.name.as_str(), a.observed))
        .collect();
    record::log_scoped(&mut rv, &observed);
    for a in &anchors {
        record::log_verdict(&mut rv, ANCHOR_EVENT, &a.name, a);
    }

    let bench_total: usize = bench_comparisons.iter().map(|c| c.n_total).sum();
    let bench_aligned: usize = bench_comparisons.iter().map(|c| c.n_aligned).sum();
    let n_pass = anchors.iter().filter(|a| a.pass).count();
    record::log_scoped(
        &mut rv,
        &[
            ("anchors_passed", n_pass as f64),
            ("anchors_total", anchors.len() as f64),
            ("bench_aligned", bench_aligned as f64),
            ("bench_total", bench_total as f64),
        ],
    );

    // --- コンソール出力 ---
    println!("--- Table 3: hybrid vs pure-abm (dataset={table3_dataset}) ---");
    println!(
        "{:<18} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "condition", "Bias", "Div.", "Pol.", "Mob.", "Mob-gain", "LLM"
    );
    for c in &table3_cells {
        println!(
            "{:<18} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>10.3} {:>8.1}",
            c.label,
            c.mean_final_bias,
            c.mean_final_diversity,
            c.mean_final_polarization,
            c.mean_final_mobilization,
            c.mean_mobilization_gain,
            c.mean_llm_calls,
        );
    }
    println!("--- Table 2: SoMoSiMu-Bench 照合 (pure-abm BC; 合成参照) ---");
    for c in &bench_comparisons {
        println!(
            "  {:<6} [{}] {}/{} 指標が整合 (source={})",
            c.movement,
            if c.n_aligned * 2 >= c.n_total {
                "OK "
            } else {
                "off"
            },
            c.n_aligned,
            c.n_total,
            c.reference_source
        );
        for row in &c.rows {
            println!(
                "    {:<20} obs={:>7.3} ref={:>7.3} |Δ|={:>6.3} tol={:>5.3} {}",
                row.metric,
                row.observed,
                row.reference,
                row.abs_error,
                row.tolerance,
                if row.aligned { "✓" } else { "·" },
            );
        }
    }
    println!("--- 論文知見アンカー (Table 2/3) ---");
    for a in &anchors {
        let hi = match a.target_hi {
            Some(h) => format!("{h:.3}"),
            None => "∞".to_string(),
        };
        println!(
            "[{}] {:<32} obs={:.4} target=[{:.3},{}]",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
        );
    }
    println!("-----------------------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, anchors.len());
    println!("{}/{} bench 指標が整合帯", bench_aligned, bench_total);

    // stage は finish() より先に閉じる (manifest.csv は finish() で封をされる)．
    if let Some(stage) = abm_trials {
        stage.close();
    }
    if let Some(cell) = &decisions.0 {
        close_shared(cell);
    }

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("条件別メトリクス → {}/metrics.csv", dir.display());
    println!("判定 (アンカー・bench) → {}/events.jsonl", dir.display());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
