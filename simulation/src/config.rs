//! シミュレーション設定．
//!
//! Mou et al. (2024) "HiSim" のコアモデル (2 階層ハイブリッド = LLM コア + ABM
//! 周辺，社会ネットワーク上の意見力学) と感度分析パラメータを保持する [`Config`]
//! と，その JSON シリアライズ表現を定義する．ABM 種別・ネットワーク種別・LLM 設定
//! などの列挙型もここに集約する．

use serde::Serialize;

pub use crate::llm::{parse_stance_mode, StanceMode};

// --------------------------------------------------------------------------- //
// ABM 種別 (周辺層の意見力学モデル)
// --------------------------------------------------------------------------- //

/// 周辺 (一般) ユーザの態度更新に用いる意見力学 ABM．
///
/// 論文付録 A の 4 モデル: Bounded Confidence (BC, Deffuant)・HK (複数ソース
/// 平均)・Social Judgement (SJ, 反発力)・Lorenz (同化 + 強化 + 分極化)．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbmModel {
    /// Bounded Confidence (Deffuant et al. 2000)．信頼境界内のみ同化 → 合意形成．
    Bc,
    /// Hegselmann–Krause (2002)．信頼境界内の複数ソース平均へ移動．
    Hk,
    /// Social Judgement．受容域内は同化，拒否域では反発 (態度から遠ざかる)．
    Sj,
    /// Lorenz (2021)．同化 + 強化 + 分極化 (極端な意見を増幅) → 二極化．
    Lorenz,
}

impl AbmModel {
    /// 短い識別ラベル (CLI / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            AbmModel::Bc => "bc",
            AbmModel::Hk => "hk",
            AbmModel::Sj => "sj",
            AbmModel::Lorenz => "lorenz",
        }
    }
}

/// 文字列から [`AbmModel`] をパースする．
pub fn parse_abm(s: &str) -> Result<AbmModel, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "bc" | "deffuant" | "bounded-confidence" => Ok(AbmModel::Bc),
        "hk" | "hegselmann-krause" => Ok(AbmModel::Hk),
        "sj" | "social-judgement" | "social-judgment" => Ok(AbmModel::Sj),
        "lorenz" => Ok(AbmModel::Lorenz),
        _ => Err(format!(
            "不正な ABM 種別: \"{}\" (bc / hk / sj / lorenz)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// ネットワーク種別
// --------------------------------------------------------------------------- //

/// 社会ネットワークの生成器種別．
///
/// `Ba` (Barabási–Albert) はスケールフリー (Pareto 分布と整合; 既定)，`Ws`
/// (Watts–Strogatz) は small-world，`Er` (Erdős–Rényi) はランダムグラフ．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkKind {
    /// Barabási–Albert (スケールフリー; 既定)．
    Ba,
    /// Watts–Strogatz (small-world)．
    Ws,
    /// Erdős–Rényi (ランダムグラフ)．
    Er,
}

impl NetworkKind {
    /// 短い識別ラベル．
    pub fn label(&self) -> &'static str {
        match self {
            NetworkKind::Ba => "ba",
            NetworkKind::Ws => "ws",
            NetworkKind::Er => "er",
        }
    }
}

/// 文字列から [`NetworkKind`] をパースする．
pub fn parse_network(s: &str) -> Result<NetworkKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "ba" | "barabasi-albert" | "barabasi" | "scale-free" => Ok(NetworkKind::Ba),
        "ws" | "watts-strogatz" | "small-world" => Ok(NetworkKind::Ws),
        "er" | "erdos-renyi" | "random" => Ok(NetworkKind::Er),
        _ => Err(format!("不正なネットワーク種別: \"{}\" (ba / ws / er)", s)),
    }
}

// --------------------------------------------------------------------------- //
// ネットワーク設定
// --------------------------------------------------------------------------- //

/// 社会ネットワーク生成のパラメータ．
#[derive(Debug, Clone, Copy)]
pub struct NetworkConfig {
    /// 生成器種別 (ba / ws / er)．
    pub kind: NetworkKind,
    /// BA の新規ノードあたりの結合数 m．
    pub ba_m: usize,
    /// WS の各ノードの近傍数 k (偶数)．
    pub ws_k: usize,
    /// WS の張り替え確率 β ∈ [0,1]．
    pub ws_beta: f64,
    /// ER の辺確率 p ∈ [0,1]．
    pub er_p: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        NetworkConfig {
            kind: NetworkKind::Ba,
            ba_m: 4,
            ws_k: 6,
            ws_beta: 0.1,
            er_p: 0.02,
        }
    }
}

// --------------------------------------------------------------------------- //
// ABM パラメータ
// --------------------------------------------------------------------------- //

/// 周辺 ABM 層の態度更新パラメータ (論文付録 A 相当)．
#[derive(Debug, Clone, Copy)]
pub struct AbmParams {
    /// 意見力学モデル種別．
    pub model: AbmModel,
    /// 社会的影響の強さ α ∈ [0,1] (同化率)．
    pub alpha: f64,
    /// 信頼境界 (confidence bound) ε．|Δ態度| < ε で同化が起きる．
    pub epsilon: f64,
    /// SJ の拒否域しきい値 (これより遠いと反発する)．
    pub rejection: f64,
    /// SJ/Lorenz の反発・分極化の強さ．
    pub repulsion: f64,
}

impl Default for AbmParams {
    fn default() -> Self {
        AbmParams {
            model: AbmModel::Bc,
            alpha: 0.3,
            epsilon: 0.4,
            rejection: 0.8,
            repulsion: 0.15,
        }
    }
}

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// 定義は `socsim-llm` に集約済み (各 replication で同一だった struct を統合)．
/// `crate::config::LlmSettings` パスは re-export で温存する．
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
#[derive(Debug, Clone)]
pub struct Config {
    /// データセット名 (metoo / roe / blm; トリガーイベントの文脈に使う)．
    pub dataset: String,
    /// エージェント数 N (= ノード数)．
    pub n_agents: usize,
    /// コア層 (LLM 駆動) の比率 ∈ [0,1]．0.0 = 純粋 ABM (LLM 呼び出し無し)．
    pub core_ratio: f64,
    /// タイムステップ数 T (論文は 14)．
    pub steps: usize,
    /// ネットワーク設定．
    pub network: NetworkConfig,
    /// ABM (周辺層) パラメータ．
    pub abm: AbmParams,
    /// 動員 (mobilization) 判定の態度しきい値 (|態度| >= this で動員とみなす)．
    pub mobilization_threshold: f64,
    /// 1 実行あたりの最大 LLM 呼び出し数 (超過分は do-nothing へフォールバック)．
    pub llm_budget: usize,
    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
    /// コア post の stance → 態度 写像モード (既定: 決定論的; `Llm` で外部 LLM 注釈)．
    pub stance: StanceMode,
    /// 結果出力ディレクトリ．
    pub output_dir: String,
}

impl Default for Config {
    /// 標準設定 (metoo, N=1000, core-ratio=0.3, T=14, BA)．
    fn default() -> Self {
        Config {
            dataset: "metoo".to_string(),
            n_agents: 1000,
            core_ratio: 0.3,
            steps: 14,
            network: NetworkConfig::default(),
            abm: AbmParams::default(),
            mobilization_threshold: 0.5,
            llm_budget: 5000,
            seed: Some(42),
            llm: LlmSettings::default(),
            stance: StanceMode::default(),
            output_dir: "results".to_string(),
        }
    }
}

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub dataset: String,
    pub n_agents: usize,
    pub core_ratio: f64,
    pub steps: usize,
    pub abm: String,
    pub alpha: f64,
    pub epsilon: f64,
    pub rejection: f64,
    pub repulsion: f64,
    pub network: String,
    pub ba_m: usize,
    pub ws_k: usize,
    pub ws_beta: f64,
    pub er_p: f64,
    pub mobilization_threshold: f64,
    pub llm_budget: usize,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub stance: String,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            dataset: self.dataset.clone(),
            n_agents: self.n_agents,
            core_ratio: self.core_ratio,
            steps: self.steps,
            abm: self.abm.model.label().to_string(),
            alpha: self.abm.alpha,
            epsilon: self.abm.epsilon,
            rejection: self.abm.rejection,
            repulsion: self.abm.repulsion,
            network: self.network.kind.label().to_string(),
            ba_m: self.network.ba_m,
            ws_k: self.network.ws_k,
            ws_beta: self.network.ws_beta,
            er_p: self.network.er_p,
            mobilization_threshold: self.mobilization_threshold,
            llm_budget: self.llm_budget,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            stance: self.stance.label().to_string(),
            output_dir: self.output_dir.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_abm_variants() {
        assert_eq!(parse_abm("bc").unwrap(), AbmModel::Bc);
        assert_eq!(parse_abm("HK").unwrap(), AbmModel::Hk);
        assert_eq!(parse_abm("sj").unwrap(), AbmModel::Sj);
        assert_eq!(parse_abm("lorenz").unwrap(), AbmModel::Lorenz);
        assert!(parse_abm("xyz").is_err());
    }

    #[test]
    fn parse_network_variants() {
        assert_eq!(parse_network("ba").unwrap(), NetworkKind::Ba);
        assert_eq!(parse_network("ws").unwrap(), NetworkKind::Ws);
        assert_eq!(parse_network("er").unwrap(), NetworkKind::Er);
        assert!(parse_network("foo").is_err());
    }
}
