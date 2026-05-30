//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で網生成・態度
//!   初期化・階層割当の init RNG を，`derive_seed(root, &[1])` で engine RNG
//!   (= scheduler + 周辺 ABM の確率的選択) を派生する．bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き Ollama→OpenAI
//!   フォールバッククライアントに閉じ込め，`temperature=0`/`seed` 固定 + プロンプト
//!   →応答キャッシュで擬似決定論化する．モデル・endpoint・温度・seed・cache-hit を
//!   `run_metadata.json` に記録する．
//!
//! # 2 階層割当
//!
//! 初期化時にネットワーク次数上位 `ceil(core_ratio · N)` 体をコア層 (LLM 駆動) と
//! し，残りを周辺層 (ABM 駆動) とする (論文の «少数の活発なオピニオンリーダ» に
//! 対応; スケールフリー網では高次数 = 影響力大)．`core_ratio = 0.0` ならコアは
//! 0 体で，純粋 ABM (LLM 呼び出し無し) になる．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufWriter;
use std::rc::Rc;

use csv::Writer;
use rand::Rng;
use serde::Serialize;

use socsim_core::{derive_seed, AgentId, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::MetadataCollector;
use socsim_net::SocialNetwork;

use crate::config::{Config, NetworkKind};
use crate::llm::{build_live_client, HiSimClient};
use crate::mechanisms::{
    AggregateMechanism, DecisionMechanism, EnvironmentMechanism, MobilizationMechanism,
    SharedBudget, SharedClient, SharedMetadata,
};
use crate::metrics::StepMetrics;
use crate::world::{CoreState, HiSimWorld, Tier};

/// 網生成・態度初期化・階層割当用 RNG ラベル．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン (= scheduler + 周辺 ABM の確率的選択) 用 RNG ラベル．
const RNG_ENGINE: u64 = 1;

/// コア層プロフィールの語彙テンプレート (決定論的割当)．
const ROLES: [&str; 6] = [
    "a journalist covering social issues with a large following",
    "an activist organizing community events",
    "a public intellectual who debates online",
    "a celebrity influencer sharing personal views",
    "a skeptical commentator who questions narratives",
    "a community leader connecting many groups",
];

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// 各タイムステップ (t=0 を含む) の集団メトリクス履歴．
    pub metrics_history: Vec<StepMetrics>,
    /// 収束したか (bias 変化が十分小さくなって停止)．
    pub converged: bool,
    /// 収束 (または最終) タイムステップ番号．
    pub final_step: usize,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名 (run_metadata 用)．
    pub llm_model: String,
    /// LLM endpoint (run_metadata 用; primary)．
    pub llm_endpoint: String,
}

/// 社会ネットワークを生成する (ba / ws / er)．
fn build_network(cfg: &Config, ids: &[AgentId], rng: &mut SimRng) -> SocialNetwork {
    match cfg.network.kind {
        NetworkKind::Ba => SocialNetwork::barabasi_albert(ids, cfg.network.ba_m.max(1), rng),
        NetworkKind::Ws => {
            SocialNetwork::watts_strogatz(ids, cfg.network.ws_k.max(2), cfg.network.ws_beta, rng)
        }
        NetworkKind::Er => SocialNetwork::erdos_renyi(ids, cfg.network.er_p, rng),
    }
}

/// 世界状態を初期化する (網構築 + 態度初期化 + 2 階層割当)．
///
/// 初期態度は init RNG から一様 `[-1, 1]` に割り当てる．ネットワーク次数上位
/// `ceil(core_ratio · N)` 体をコア層 (LLM 駆動) とし，残りを周辺層とする
/// (高次数ノード = 影響力大)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> HiSimWorld {
    let ids: Vec<AgentId> = (0..cfg.n_agents as u64).map(AgentId).collect();
    let network = build_network(cfg, &ids, rng);

    let mut attitude: BTreeMap<AgentId, f64> = BTreeMap::new();
    for &id in &ids {
        attitude.insert(id, rng.gen_range(-1.0..1.0));
    }

    // 階層割当: 次数降順 → 同点は AgentId 昇順 で上位 n_core をコアに．
    let n_core =
        ((cfg.core_ratio.clamp(0.0, 1.0) * cfg.n_agents as f64).ceil() as usize).min(cfg.n_agents);
    let mut by_degree: Vec<(usize, AgentId)> =
        ids.iter().map(|&id| (network.degree(id), id)).collect();
    by_degree.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let core_set: std::collections::BTreeSet<AgentId> =
        by_degree.iter().take(n_core).map(|(_, id)| *id).collect();

    let mut tier: BTreeMap<AgentId, Tier> = BTreeMap::new();
    let mut core: BTreeMap<AgentId, CoreState> = BTreeMap::new();
    for (idx, &id) in ids.iter().enumerate() {
        if core_set.contains(&id) {
            tier.insert(id, Tier::Core);
            let profile = ROLES[idx % ROLES.len()].to_string();
            core.insert(id, CoreState::new(profile));
        } else {
            tier.insert(id, Tier::Ordinary);
        }
    }

    HiSimWorld::new(network, attitude, tier, core, cfg.steps as u64)
}

/// シミュレーションを実行する (本番 LLM クライアントを構築して駆動)．
pub fn run(cfg: &Config) -> std::result::Result<SimulationResult, String> {
    let client =
        build_live_client(&cfg.llm).map_err(|e| format!("LLM クライアント構築に失敗: {e}"))?;
    run_with_client(cfg, client)
}

/// 与えられた [`HiSimClient`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．`core_ratio = 0.0` のときコアは
/// 0 体なので LLM クライアントは構築されても一切呼ばれない (純粋 ABM)．
pub fn run_with_client(
    cfg: &Config,
    client: HiSimClient,
) -> std::result::Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_budget: SharedBudget = Rc::new(RefCell::new(cfg.llm_budget));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(EnvironmentMechanism::new(&cfg.dataset)))
        .add_mechanism(Box::new(DecisionMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            Rc::clone(&shared_budget),
            cfg.llm.clone(),
            cfg.stance,
        )))
        .add_mechanism(Box::new(MobilizationMechanism::new(cfg.abm)))
        .add_mechanism(Box::new(AggregateMechanism::new(
            cfg.mobilization_threshold,
            1e-9,
        )))
        .build();

    let mut metrics_history: Vec<StepMetrics> = Vec::new();

    // 初期状態 (t=0) を記録．
    {
        let w = sim.world();
        metrics_history.push(StepMetrics::compute(
            &w.attitude,
            &w.tier,
            cfg.mobilization_threshold,
            0,
            0,
        ));
    }

    let mut converged = false;
    let mut final_step = 0usize;
    let threshold = cfg.mobilization_threshold;

    sim.run_observed(|report| {
        let t = report.t as usize;
        let w = report.world;
        let llm_actions = *report
            .scratch
            .get::<usize>("llm_actions")
            .unwrap_or(&0usize);
        metrics_history.push(StepMetrics::compute(
            &w.attitude,
            &w.tier,
            threshold,
            llm_actions,
            t,
        ));
        converged = report.stopped;
        final_step = t;
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let metadata = shared_meta.borrow().clone();

    Ok(SimulationResult {
        metrics_history,
        converged,
        final_step,
        metadata,
        llm_model,
        llm_endpoint,
    })
}

// --------------------------------------------------------------------------- //
// 出力
// --------------------------------------------------------------------------- //

/// メトリクス履歴を long-format CSV (metrics.csv) に保存する．
///
/// 各 [`StepMetrics`] を [`StepMetrics::to_rows`] で複数の `MetricRow` に展開して
/// から逐次 `serialize` する (1 ステップ → 指標名ごとに 1 行)．`socsim_results::
/// write_csv` は `&[T]` を 1 要素 1 行で直列化するだけでこの «1 要素を複数行へ
/// 展開» を表現できないため，本 writer は repo ローカルのまま残す．
pub fn save_metrics(metrics: &[StepMetrics], output_dir: &str) {
    let path = format!("{}/metrics.csv", output_dir);
    let file = File::create(&path).expect("metrics.csv の作成に失敗");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    for m in metrics {
        for row in m.to_rows() {
            wtr.serialize(row).expect("メトリクス行の書き込みに失敗");
        }
    }
    wtr.flush().expect("フラッシュに失敗");
}

/// `run_metadata.json` の構造体 (LLM モデル・endpoint・温度・seed・cache 統計)．
#[derive(Serialize)]
pub struct RunMetadataJson {
    pub provider: String,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub core_ratio: f64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub determinism_note: &'static str,
}

/// `run_metadata.json` を保存する．
pub fn save_run_metadata(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let provider =
        if result.llm_endpoint.contains("11434") || result.llm_endpoint.contains("ollama") {
            "ollama"
        } else if result.llm_endpoint.contains("mock") {
            "mock"
        } else {
            "openai"
        };
    let meta = RunMetadataJson {
        provider: provider.to_string(),
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        core_ratio: cfg.core_ratio,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (with temperature=0 and fixed seed) is the reproducibility \
                           mechanism. The socsim core (network generation, tier assignment, the \
                           Ordinary-tier ABM opinion dynamics, and the scheduler) is deterministic \
                           given the seed. With core-ratio 0.0 there are no LLM calls (pure ABM).",
    };
    // pretty-print JSON の書き出しは socsim_results::write_json に委譲する
    // (内部は serde_json::to_writer_pretty + flush; 従来の writer とバイト等価)．
    // provider/model/endpoint/temperature/seed/core_ratio の値は従来どおり
    // result / cfg から採り，RunMetadataJson の構造 (フィールド名・順序・
    // determinism_note) を保持する (`MetadataCollector::summary()` は cache-hit
    // 100%% 再実行や呼び出し 0 件で endpoint/model が変わりうるため，バイト等価
    // のためここでは使わない)．
    let path = format!("{}/run_metadata.json", output_dir);
    socsim_results::write_json(&meta, &path).expect("run_metadata.json の書き込みに失敗");
}

/// 出力ディレクトリを作成する．
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("出力ディレクトリの作成に失敗");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AbmModel, AbmParams, LlmSettings, NetworkConfig};
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    /// 発信行動を返す mock コアクライアント．
    fn scripted_client() -> HiSimClient {
        let backend = ScriptedClient::new("mock-llama3.2", |_prompt: &str| {
            "THOUGHT: I will speak up.\nACTION: post\nMESSAGE: I support this movement and stand in solidarity.".to_string()
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn cfg_with(core_ratio: f64, model: AbmModel) -> Config {
        Config {
            dataset: "metoo".to_string(),
            n_agents: 80,
            core_ratio,
            steps: 14,
            network: NetworkConfig::default(),
            abm: AbmParams {
                model,
                ..AbmParams::default()
            },
            mobilization_threshold: 0.5,
            llm_budget: 10_000,
            seed: Some(42),
            llm: LlmSettings::default(),
            stance: crate::config::StanceMode::default(),
            output_dir: "results".to_string(),
        }
    }

    #[test]
    fn pure_abm_makes_no_llm_calls() {
        let cfg = cfg_with(0.0, AbmModel::Bc);
        let r = run_with_client(&cfg, scripted_client()).unwrap();
        assert_eq!(r.metadata.total(), 0, "core-ratio 0.0 must not call LLM");
        assert_eq!(r.metrics_history[0].t, 0);
    }

    #[test]
    fn pure_abm_is_deterministic() {
        let cfg = cfg_with(0.0, AbmModel::Bc);
        let a = run_with_client(&cfg, scripted_client()).unwrap();
        let b = run_with_client(&cfg, scripted_client()).unwrap();
        let af: Vec<f64> = a.metrics_history.iter().map(|m| m.macro_bias).collect();
        let bf: Vec<f64> = b.metrics_history.iter().map(|m| m.macro_bias).collect();
        assert_eq!(af, bf);
        assert_eq!(a.final_step, b.final_step);
    }

    #[test]
    fn tier_assignment_respects_core_ratio() {
        let cfg = cfg_with(0.3, AbmModel::Bc);
        let mut rng = SimRng::from_seed(derive_seed(42, &[RNG_WORLD_INIT]));
        let world = init_world(&cfg, &mut rng);
        let expected = (0.3 * 80.0_f64).ceil() as usize;
        assert_eq!(world.n_core(), expected);
    }

    #[test]
    fn hybrid_calls_llm() {
        let cfg = cfg_with(0.3, AbmModel::Bc);
        let r = run_with_client(&cfg, scripted_client()).unwrap();
        assert!(
            r.metadata.total() > 0,
            "hybrid should call LLM for core tier"
        );
    }

    #[test]
    fn stance_default_is_bit_identical_to_deterministic() {
        // 既定 (StanceMode::Deterministic) は従来挙動と完全一致するべき．
        let mut cfg = cfg_with(0.3, AbmModel::Bc);
        assert_eq!(cfg.stance, crate::config::StanceMode::Deterministic);
        let baseline = run_with_client(&cfg, scripted_client()).unwrap();

        cfg.stance = crate::config::StanceMode::Deterministic;
        let again = run_with_client(&cfg, scripted_client()).unwrap();

        let a: Vec<f64> = baseline
            .metrics_history
            .iter()
            .map(|m| m.macro_bias)
            .collect();
        let b: Vec<f64> = again.metrics_history.iter().map(|m| m.macro_bias).collect();
        assert_eq!(a, b, "deterministic stance must be bit-identical");
        // 決定論経路は post 分類で追加 LLM 呼び出しをしない (= core 数 × steps のみ)．
        assert_eq!(baseline.metadata.total(), again.metadata.total());
    }

    #[test]
    fn stance_llm_mode_adds_annotation_calls() {
        // 外部 LLM stance 注釈は post ごとに追加の LLM 呼び出しを発生させる．
        let det = run_with_client(&cfg_with(0.3, AbmModel::Bc), scripted_client()).unwrap();
        let mut cfg = cfg_with(0.3, AbmModel::Bc);
        cfg.stance = crate::config::StanceMode::Llm;
        let llm = run_with_client(&cfg, scripted_client()).unwrap();
        assert!(
            llm.metadata.total() > det.metadata.total(),
            "LLM stance mode should make extra annotation calls: {} vs {}",
            llm.metadata.total(),
            det.metadata.total()
        );
    }

    #[test]
    fn all_four_abm_models_run() {
        for model in [AbmModel::Bc, AbmModel::Hk, AbmModel::Sj, AbmModel::Lorenz] {
            let cfg = cfg_with(0.0, model);
            let r = run_with_client(&cfg, scripted_client()).unwrap();
            assert!(
                r.metrics_history.len() > 1,
                "{:?} should produce steps",
                model
            );
        }
    }

    #[test]
    fn bc_vs_lorenz_diverge() {
        // BC は合意 (低分極) 寄り，Lorenz は二極化 (高分極) 寄り (同一初期条件)．
        let bc = run_with_client(&cfg_with(0.0, AbmModel::Bc), scripted_client()).unwrap();
        let lz = run_with_client(&cfg_with(0.0, AbmModel::Lorenz), scripted_client()).unwrap();
        let bc_pol = bc.metrics_history.last().unwrap().polarization;
        let lz_pol = lz.metrics_history.last().unwrap().polarization;
        assert!(
            lz_pol >= bc_pol,
            "Lorenz polarization {lz_pol} should be >= BC polarization {bc_pol}"
        );
    }
}
