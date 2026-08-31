//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (runvault の run ディレクトリ) と Python 可視化を検証する
//! ための補助バイナリ．`socsim-llm::mock::ScriptedClient` で決定論的にコア層の
//! 行動を駆動し，本番 `run` と同じ経路で結果を記録する．`core_ratio > 0` を
//! 指定するのでハイブリッド経路 (LLM コア + ABM 周辺) を ScriptedClient 経由で
//! 動かせる．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use runvault::{Run, RunOptions};

use hisim_simulation::config::{AbmModel, AbmParams, Config, LlmSettings, NetworkConfig};
use hisim_simulation::llm::wrap_client;
use hisim_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use hisim_simulation::simulation::run_with_client;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};

/// シードは固定 (スモークの目的は «同じ入力で同じ出力» の確認)．
const SEED: u64 = 42;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());

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
        seed: Some(SEED),
        llm: LlmSettings::default(),
        stance: hisim_simulation::config::StanceMode::default(),
    };

    // コア層擬似挙動: 支持メッセージを発信し続ける (call-to-action を周辺へ伝播)．
    let backend = ScriptedClient::new("mock-llama3.2", |_prompt: &str| {
        "THOUGHT: I will speak up.\nACTION: post\nMESSAGE: I support this movement and we must stand in solidarity for justice.".to_string()
    });
    let client = wrap_client(backend, PromptCache::in_memory());
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let parameters = cfg.to_run_config_json(SEED);
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&base)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(SEED)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, client).expect("mock run failed");
    record::log_simulation(&mut rv, &result);

    let last = result.metrics_history.last().unwrap();
    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("mock smoke wrote: {}", dir.display());
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
