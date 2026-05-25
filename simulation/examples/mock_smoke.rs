//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (metrics.csv / run_metadata.json / config.json) と Python
//! 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で
//! 決定論的にコア層の行動を駆動し，本番 `run` と同じ writer で結果を書き出す．
//! `core_ratio > 0` を指定するためハイブリッド経路 (LLM コア + ABM 周辺) を
//! ScriptedClient 経由で動かせる．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;
use std::fs;

use chrono::Local;

use hisim_simulation::config::{AbmModel, AbmParams, Config, LlmSettings, NetworkConfig};
use hisim_simulation::llm::wrap_client;
use hisim_simulation::simulation::{
    ensure_output_dir, run_with_client, save_metrics, save_run_metadata,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        dataset: "metoo".to_string(),
        n_agents: 200,
        core_ratio: 0.3,
        steps: 14,
        network: NetworkConfig::default(),
        abm: AbmParams {
            model: AbmModel::Bc,
            ..AbmParams::default()
        },
        mobilization_threshold: 0.5,
        llm_budget: 10_000,
        seed: Some(42),
        llm: LlmSettings::default(),
        output_dir: output_dir.clone(),
    };

    // コア層擬似挙動: 支持メッセージを発信し続ける (call-to-action を周辺へ伝播)．
    let backend = ScriptedClient::new("mock-llama3.2", |_prompt: &str| {
        "THOUGHT: I will speak up.\nACTION: post\nMESSAGE: I support this movement and we must stand in solidarity for justice.".to_string()
    });
    let client = wrap_client(backend, PromptCache::in_memory());

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {output_dir}");
    println!(
        "final bias={:.4} diversity={:.4} mobilized={} polarization={:.4} core_influence={:.4} steps={}",
        last.macro_bias,
        last.macro_diversity,
        last.mobilized,
        last.polarization,
        last.core_influence,
        result.final_step
    );
    println!(
        "LLM calls: {} (mock; cache-hit {:.1}%)",
        result.metadata.total(),
        result.metadata.cache_hit_rate() * 100.0
    );
}
