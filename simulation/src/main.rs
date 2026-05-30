//! Mou et al. (2024) "Unveiling the Truth and Facilitating Change: Towards
//! Agent-based Large-scale Social Movement Simulation" (HiSim) — 再現実験の CLI
//! エントリポイント．
//!
//! `run`       : 単一設定で 2 階層ハイブリッド (LLM コア + ABM 周辺) を実行する．
//!               `--core-ratio 0.0` なら純粋 ABM (LLM 呼び出し無し)．
//! `sweep`     : コア比率 × ABM 種別 × ネットワーク構造 を走査し，最終マクロ指標を
//!               `sweep_summary.csv` に集計する．
//! `reproduce` : 論文 Table 2/3 の見出し的知見 (ハイブリッド vs 純 ABM) と
//!               SoMoSiMu-Bench 照合を一括再現し reproduce_summary.json + 図に集計する．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use hisim_simulation::bench::{compare_to_bench, reference_curve, MovementMetrics};
use hisim_simulation::config::{
    parse_abm, parse_network, parse_stance_mode, AbmModel, AbmParams, Config, LlmSettings,
    NetworkConfig, NetworkKind, StanceMode,
};
use hisim_simulation::reproduce_mock::build_reproduce_client;
use hisim_simulation::simulation::{
    ensure_output_dir, run, run_with_client, save_metrics, save_run_metadata, SimulationResult,
};

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

/// `sweep_summary.csv` の 1 行．
#[derive(serde::Serialize)]
struct SweepRow {
    dataset: String,
    abm: String,
    network: String,
    core_ratio: f64,
    n_agents: usize,
    run: usize,
    seed: u64,
    converged: bool,
    final_step: usize,
    final_macro_bias: f64,
    final_macro_diversity: f64,
    final_mobilized: usize,
    final_polarization: f64,
    final_core_influence: f64,
    total_llm_calls: usize,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    dataset: String,
    core_ratio_values: Vec<f64>,
    abm_values: Vec<String>,
    network_values: Vec<String>,
    n_agents: usize,
    steps: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// 派生シードのラベルに使う文字列ハッシュ (explicit identity)．
fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
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

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

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
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        stance,
        output_dir: output_dir.clone(),
    };

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&cfg.output_dir);

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
        "seed: {:?} | llm-budget: {} | stance: {} | LLM: temp={} llm_seed={} cache={}",
        cfg.seed,
        cfg.llm_budget,
        cfg.stance.label(),
        cfg.llm.temperature,
        cfg.llm.seed,
        args.cache_path
    );
    println!("出力先: {}", cfg.output_dir);
    println!("-----------------------------------------------------------------");

    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json (pretty-print JSON; socsim_results::write_json に委譲)．
    {
        let path = format!("{}/config.json", cfg.output_dir);
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
    }

    // latest シンボリックリンクを再作成する (best-effort; 従来同様エラーは無視)．
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

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
    println!("メトリクス → {}/metrics.csv", cfg.output_dir);
    println!("LLM メタ   → {}/run_metadata.json", cfg.output_dir);
    println!("設定       → {}/config.json", cfg.output_dir);
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

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = core_ratio_values.len() * abm_models.len() * net_kinds.len() * args.runs;

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
    println!("出力先: {}", sweep_dir);
    println!("-----------------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &net_kind in &net_kinds {
        for &abm_model in &abm_models {
            for &core_ratio in &core_ratio_values {
                for run_idx in 0..args.runs {
                    let seed = socsim_core::derive_seed(
                        args.seed,
                        &[
                            label_hash(net_kind.label()),
                            label_hash(abm_model.label()),
                            (core_ratio * 1000.0) as u64,
                            run_idx as u64,
                        ],
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
                        llm: LlmSettings {
                            temperature: args.llm_temperature,
                            seed: args.llm_seed,
                            cache_path: Some(args.cache_path.clone()),
                        },
                        stance: StanceMode::default(),
                        output_dir: sweep_dir.clone(),
                    };

                    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                    let last = result.metrics_history.last().unwrap();

                    summary_rows.push(SweepRow {
                        dataset: args.dataset.clone(),
                        abm: abm_model.label().to_string(),
                        network: net_kind.label().to_string(),
                        core_ratio,
                        n_agents: args.n_agents,
                        run: run_idx,
                        seed,
                        converged: result.converged,
                        final_step: result.final_step,
                        final_macro_bias: last.macro_bias,
                        final_macro_diversity: last.macro_diversity,
                        final_mobilized: last.mobilized,
                        final_polarization: last.polarization,
                        final_core_influence: last.core_influence,
                        total_llm_calls: result.metadata.total(),
                        cache_hit_rate: result.metadata.cache_hit_rate(),
                    });

                    done += 1;
                }
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

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            dataset: args.dataset.clone(),
            core_ratio_values: core_ratio_values.clone(),
            abm_values: abm_models.iter().map(|m| m.label().to_string()).collect(),
            network_values: net_kinds.iter().map(|n| n.label().to_string()).collect(),
            n_agents: args.n_agents,
            steps: args.steps,
            runs: args.runs,
            seed: args.seed,
            llm_temperature: args.llm_temperature,
            llm_seed: args.llm_seed,
        };
        let path = format!("{}/sweep_config.json", sweep_dir);
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_sweep", timestamp));

    println!("=================================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("ABM 種別別の平均 分極化 polarization:");
    for &abm_model in &abm_models {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| r.abm == abm_model.label())
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg = rows.iter().map(|r| r.final_polarization).sum::<f64>() / rows.len() as f64;
        println!("  abm={:<7} → polarization̄ = {:.4}", abm_model.label(), avg);
    }
    println!("-----------------------------------------------------------------");
    println!("サマリ → {}/sweep_summary.csv", sweep_dir);
    println!("設定   → {}/sweep_config.json", sweep_dir);
}

// ---------------------------------------------------------------------------
// reproduce — Table 2/3 + SoMoSiMu-Bench 照合
// ---------------------------------------------------------------------------

/// 1 レジーム (hybrid / pure-abm) × ABM を `runs` 回回した集計セル (Table 3 の元)．
#[derive(serde::Serialize, Clone)]
struct ReproCell {
    /// 条件ラベル (例: "hybrid_bc" / "pureabm_lorenz")．
    label: String,
    /// レジーム ("hybrid" = LLM コア + ABM 周辺 / "pure-abm" = core-ratio 0)．
    regime: String,
    /// 周辺 ABM 種別．
    abm: String,
    /// コア比率 (hybrid は core_ratio，pure-abm は 0.0)．
    core_ratio: f64,
    runs: usize,
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

/// 観測値と論文の定性的知見を突き合わせた 1 アンカー．
#[derive(serde::Serialize)]
struct ReproAnchor {
    name: String,
    paper: String,
    observed: f64,
    target_lo: f64,
    target_hi: f64,
    pass: bool,
}

/// 1 レジーム × ABM を `runs` 回実行して集計セルを作る (代表 run の履歴を CSV 保存)．
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
    out_dir: &str,
) -> (ReproCell, Vec<MovementMetrics>) {
    let mut final_bias = 0.0;
    let mut final_div = 0.0;
    let mut final_pol = 0.0;
    let mut final_mob = 0.0;
    let mut mob_gain = 0.0;
    let mut llm_calls = 0.0;
    let mut movement_runs: Vec<MovementMetrics> = Vec::with_capacity(runs);
    let mut representative: Option<Vec<hisim_simulation::metrics::StepMetrics>> = None;

    for run_idx in 0..runs {
        let seed = socsim_core::derive_seed(
            root_seed,
            &[
                label_hash(regime),
                label_hash(abm_model.label()),
                label_hash(dataset),
                run_idx as u64,
            ],
        );
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
            output_dir: out_dir.to_string(),
        };

        // pure-abm (core_ratio 0) は LLM を一切呼ばないので mock も live も不要．
        let result: SimulationResult = if core_ratio == 0.0 {
            run_with_client(&cfg, build_reproduce_client())
                .unwrap_or_else(|e| panic!("実行に失敗 ({label}): {e}"))
        } else if mock {
            run_with_client(&cfg, build_reproduce_client())
                .unwrap_or_else(|e| panic!("mock 実行に失敗 ({label}): {e}"))
        } else {
            run(&cfg).unwrap_or_else(|e| panic!("実行に失敗 ({label}): {e}"))
        };

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
            representative = Some(result.metrics_history.clone());
        }
    }

    if let Some(hist) = representative {
        // 代表 run (run 0) の long-format メトリクスを条件別名で書き出す
        // (Python 側で時系列描画に使う; save_metrics は metrics.csv 固定名なので使わない)．
        let path = format!("{out_dir}/metrics_{label}.csv");
        let file = std::fs::File::create(&path).expect("metrics_<label>.csv の作成に失敗");
        let mut wtr = csv::Writer::from_writer(std::io::BufWriter::new(file));
        for m in &hist {
            for row in m.to_rows() {
                wtr.serialize(row).expect("メトリクス行の書き込みに失敗");
            }
        }
        wtr.flush().expect("フラッシュに失敗");
    }

    let n = runs.max(1) as f64;
    let cell = ReproCell {
        label: label.to_string(),
        regime: regime.to_string(),
        abm: abm_model.label().to_string(),
        core_ratio,
        runs,
        mean_final_bias: final_bias / n,
        mean_final_diversity: final_div / n,
        mean_final_polarization: final_pol / n,
        mean_final_mobilization: final_mob / n,
        mean_mobilization_gain: mob_gain / n,
        mean_llm_calls: llm_calls / n,
    };
    (cell, movement_runs)
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

    let ts = timestamp();
    let out_dir = format!("{}/reproduce_{}", args.output_dir, ts);
    ensure_output_dir(&out_dir);
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
    println!("出力先: {out_dir}");
    println!("-----------------------------------------------------------------");

    // --- Table 3: hybrid vs pure-abm の ABM 別行列 (代表 dataset = 先頭) ---
    // 純 ABM 経路 (core-ratio 0) は LLM 0 呼び出しで完全決定論的 = オフライン検証経路．
    let table3_dataset = datasets
        .first()
        .cloned()
        .unwrap_or_else(|| "metoo".to_string());
    let mut table3_cells: Vec<ReproCell> = Vec::new();
    // bench 照合用に «pure-abm BC» の運動指標を dataset ごとに集める．
    for &abm_model in &abm_models {
        for &(regime, ratio) in &[("pure-abm", 0.0), ("hybrid", args.core_ratio)] {
            let label = format!("{}_{}", regime.replace('-', ""), abm_model.label());
            let (cell, _runs) = run_repro_cell(
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
                &out_dir,
            );
            table3_cells.push(cell);
        }
    }

    // --- Table 2: SoMoSiMu-Bench 照合 (dataset 別; pure-abm BC を観測系列とする) ---
    // 純 ABM の動員ダイナミクスを各運動の較正済み合成参照と照合する (オフライン経路)．
    let bench_abm = AbmModel::Bc;
    let mut bench_comparisons: Vec<hisim_simulation::bench::BenchComparison> = Vec::new();
    for ds in &datasets {
        let label = format!("bench_{ds}");
        let (_cell, movement_runs) = run_repro_cell(
            &label, "pure-abm", bench_abm, 0.0, net_kind, ds, n_agents, steps, stance, runs,
            args.seed, args.mock, &llm, &out_dir,
        );
        let observed = mean_movement(&movement_runs);
        let reference = reference_curve(ds, steps);
        bench_comparisons.push(compare_to_bench(ds, observed, &reference));
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
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: f64| {
        anchors.push(ReproAnchor {
            name: name.to_string(),
            paper: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && obs <= hi,
        });
    };

    // T3-A: pure-abm は LLM を一切呼ばない (オフライン検証可能経路)．
    push(
        "pure_abm_zero_llm_calls",
        "core-ratio 0 = no LLM",
        bc_pure.mean_llm_calls,
        0.0,
        0.0,
    );
    // T3-B: 二極化の順序 BC ≤ {SJ, Lorenz} (論文 §A: BC/HK は合意，SJ/Lorenz は分極)．
    push(
        "polarization_lorenz>=bc",
        "Lorenz polarizes vs BC consensus",
        lorenz_pure.mean_final_polarization - bc_pure.mean_final_polarization,
        -1e-9,
        f64::INFINITY,
    );
    push(
        "polarization_sj>=bc",
        "SJ polarizes vs BC consensus",
        sj_pure.mean_final_polarization - bc_pure.mean_final_polarization,
        -1e-9,
        f64::INFINITY,
    );
    // T3-C: BC/HK は合意寄り (低分極; 分極 < 0.5)．
    push(
        "bc_low_polarization",
        "BC reaches consensus (low polarization)",
        bc_pure.mean_final_polarization,
        0.0,
        0.5,
    );
    push(
        "hk_low_polarization",
        "HK reaches consensus (low polarization)",
        hk_pure.mean_final_polarization,
        0.0,
        0.5,
    );
    // T3-D: ハイブリッド (LLM コア) は純 ABM より動員を牽引する
    //   (mock では支持コアが call-to-action を発信 → 動員の伸びが純 ABM 以上)．
    push(
        "hybrid_amplifies_mobilization (gain_hybrid - gain_pureabm >= 0)",
        "core LLM drives mobilization",
        bc_hybrid.mean_mobilization_gain - bc_pure.mean_mobilization_gain,
        -1e-9,
        f64::INFINITY,
    );

    // bench アンカー: 各運動で過半数の指標が整合帯に入る．
    let bench_total: usize = bench_comparisons.iter().map(|c| c.n_total).sum();
    let bench_aligned: usize = bench_comparisons.iter().map(|c| c.n_aligned).sum();

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
    println!("--- Table 2: SoMoSiMu-Bench 照合 (pure-abm BC; 較正済み合成参照) ---");
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
        let hi = if a.target_hi.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.3}", a.target_hi)
        };
        println!(
            "[{}] {:<58} obs={:.4} target=[{:.3},{}]",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
        );
    }
    let n_pass = anchors.iter().filter(|a| a.pass).count();
    println!("-----------------------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, anchors.len());
    println!("{}/{} bench 指標が整合帯", bench_aligned, bench_total);

    // --- reproduce_summary.json ---
    let summary = serde_json::json!({
        "timestamp": ts,
        "mode": if args.mock { "mock" } else { "live" },
        "config": {
            "datasets": datasets,
            "abm_values": abm_models.iter().map(|m| m.label()).collect::<Vec<_>>(),
            "core_ratio": args.core_ratio,
            "n_agents": n_agents,
            "steps": steps,
            "runs": runs,
            "network": net_kind.label(),
            "stance": stance.label(),
            "seed": args.seed,
        },
        "table3_hybrid_vs_pureabm": table3_cells,
        "table3_dataset": table3_dataset,
        "bench_comparisons": bench_comparisons,
        "bench_aligned": bench_aligned,
        "bench_total": bench_total,
        "anchors": anchors,
        "n_pass": n_pass,
        "n_total": anchors.len(),
    });
    let path = format!("{out_dir}/reproduce_summary.json");
    write_json(&summary, &path).expect("reproduce_summary.json の書き込みに失敗");
    let _ = refresh_latest_symlink(&args.output_dir, &format!("reproduce_{ts}"));
    println!("サマリ → {path}");
    println!("条件別メトリクス → {out_dir}/metrics_<condition>.csv");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
