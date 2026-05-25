//! Mou et al. (2024) "Unveiling the Truth and Facilitating Change: Towards
//! Agent-based Large-scale Social Movement Simulation" (HiSim) — 再現実験の CLI
//! エントリポイント．
//!
//! `run`       : 単一設定で 2 階層ハイブリッド (LLM コア + ABM 周辺) を実行する．
//!               `--core-ratio 0.0` なら純粋 ABM (LLM 呼び出し無し)．
//! `sweep`     : コア比率 × ABM 種別 × ネットワーク構造 を走査し，最終マクロ指標を
//!               `sweep_summary.csv` に集計する．
//! `reproduce` : SoMoSiMu-Bench 照合・Table 3 再現 (Phase 3; 未実装スタブ)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use hisim_simulation::config::{
    parse_abm, parse_network, AbmModel, AbmParams, Config, LlmSettings, NetworkConfig, NetworkKind,
};
use hisim_simulation::simulation::{ensure_output_dir, run, save_metrics, save_run_metadata};

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
    /// SoMoSiMu-Bench 照合・Table 3 再現 (Phase 3; 未実装)．
    Reproduce,
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
        "seed: {:?} | llm-budget: {} | LLM: temp={} llm_seed={} cache={}",
        cfg.seed, cfg.llm_budget, cfg.llm.temperature, cfg.llm.seed, args.cache_path
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
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce => {
            eprintln!(
                "reproduce は Phase 3 で実装予定です (SoMoSiMu-Bench 照合・Table 3 再現)．\
                 現状は run / sweep を使ってください．"
            );
            std::process::exit(1);
        }
    }
}
