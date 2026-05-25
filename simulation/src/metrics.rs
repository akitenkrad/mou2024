//! 評価指標 (論文 §5/§7 のマクロシステム指標に対応)．
//!
//! 各タイムステップの世界状態 (態度分布) から，集団態度の偏り・多様性・動員規模・
//! 分極化などの集団指標を計算する．
//!
//! | 指標 | 定義 | 論文での対応 |
//! |------|------|-------------|
//! | `macro_bias` | 全エージェント態度の平均 | Table 3 (ΔBias) |
//! | `macro_diversity` | 態度分布の分散 | Table 3 (ΔDiv.) |
//! | `mobilized` | |態度| >= 閾値 のエージェント数 | 動員曲線 |
//! | `polarization` | 態度分布の双峰度 (二極化) | エコーチェンバー再現 (5.5節) |
//! | `core_influence` | コア層の平均態度 (周辺へのドライバ) | スケーラビリティ (Fig. 3) |

use std::collections::BTreeMap;

use serde::Serialize;
use socsim_core::AgentId;

use crate::world::Tier;

/// 全エージレント態度の平均 (= macro_bias = 集団態度の偏り)．
pub fn macro_bias(attitude: &BTreeMap<AgentId, f64>) -> f64 {
    let n = attitude.len();
    if n == 0 {
        return 0.0;
    }
    attitude.values().sum::<f64>() / n as f64
}

/// 態度分布の分散 (= macro_diversity = 意見多様性)．
pub fn macro_diversity(attitude: &BTreeMap<AgentId, f64>) -> f64 {
    let n = attitude.len();
    if n == 0 {
        return 0.0;
    }
    let mean = macro_bias(attitude);
    attitude.values().map(|a| (a - mean).powi(2)).sum::<f64>() / n as f64
}

/// 動員された (|態度| >= threshold) エージェント数．
pub fn mobilized(attitude: &BTreeMap<AgentId, f64>, threshold: f64) -> usize {
    attitude.values().filter(|a| a.abs() >= threshold).count()
}

/// 分極化指標 (双峰度の代理): 態度が両極 (|態度| > 0.5) に二分される度合い．
///
/// 正側極端群と負側極端群がともに存在し，かつ中庸が少ないほど高い．
/// `4 · p_+ · p_-` (両極の積; 50/50 で最大 1，片寄りで 0) で近似する．
pub fn polarization(attitude: &BTreeMap<AgentId, f64>) -> f64 {
    let n = attitude.len();
    if n == 0 {
        return 0.0;
    }
    let pos = attitude.values().filter(|&&a| a > 0.5).count() as f64;
    let neg = attitude.values().filter(|&&a| a < -0.5).count() as f64;
    let nf = n as f64;
    4.0 * (pos / nf) * (neg / nf)
}

/// コア層の平均態度 (周辺層へのドライバ; core_influence の代理)．
///
/// コア層が無ければ 0．
pub fn core_influence(attitude: &BTreeMap<AgentId, f64>, tier: &BTreeMap<AgentId, Tier>) -> f64 {
    let core: Vec<f64> = attitude
        .iter()
        .filter(|(id, _)| matches!(tier.get(id), Some(Tier::Core)))
        .map(|(_, &a)| a)
        .collect();
    if core.is_empty() {
        return 0.0;
    }
    core.iter().sum::<f64>() / core.len() as f64
}

/// 1 タイムステップ分の集団指標 (metrics.csv の long-format 行へ展開する元)．
#[derive(Debug, Clone, Serialize)]
pub struct StepMetrics {
    /// タイムステップ t．
    pub t: usize,
    /// 集団態度の偏り (平均態度)．
    pub macro_bias: f64,
    /// 意見多様性 (態度分散)．
    pub macro_diversity: f64,
    /// 動員されたエージェント数．
    pub mobilized: usize,
    /// 分極化指標．
    pub polarization: f64,
    /// コア層の平均態度 (周辺へのドライバ)．
    pub core_influence: f64,
    /// 当該ステップで LLM 呼び出しを行ったコアエージェント数．
    pub llm_actions: usize,
}

impl StepMetrics {
    /// 世界状態の現スナップショットから集団指標を計算する．
    pub fn compute(
        attitude: &BTreeMap<AgentId, f64>,
        tier: &BTreeMap<AgentId, Tier>,
        mobilization_threshold: f64,
        llm_actions: usize,
        t: usize,
    ) -> Self {
        StepMetrics {
            t,
            macro_bias: macro_bias(attitude),
            macro_diversity: macro_diversity(attitude),
            mobilized: mobilized(attitude, mobilization_threshold),
            polarization: polarization(attitude),
            core_influence: core_influence(attitude, tier),
            llm_actions,
        }
    }
}

/// metrics.csv の long-format 1 行 (t, metric, value)．
#[derive(Debug, Clone, Serialize)]
pub struct MetricRow {
    /// タイムステップ t．
    pub t: usize,
    /// 指標名．
    pub metric: String,
    /// 値．
    pub value: f64,
}

impl StepMetrics {
    /// long-format 行の列へ展開する．
    pub fn to_rows(&self) -> Vec<MetricRow> {
        let pairs: [(&str, f64); 6] = [
            ("macro_bias", self.macro_bias),
            ("macro_diversity", self.macro_diversity),
            ("mobilized", self.mobilized as f64),
            ("polarization", self.polarization),
            ("core_influence", self.core_influence),
            ("llm_actions", self.llm_actions as f64),
        ];
        pairs
            .iter()
            .map(|&(name, v)| MetricRow {
                t: self.t,
                metric: name.to_string(),
                value: v,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(vals: &[f64]) -> BTreeMap<AgentId, f64> {
        vals.iter()
            .enumerate()
            .map(|(i, &v)| (AgentId(i as u64), v))
            .collect()
    }

    #[test]
    fn bias_is_mean() {
        let d = dist(&[0.2, 0.4, 0.6]);
        assert!((macro_bias(&d) - 0.4).abs() < 1e-12);
    }

    #[test]
    fn diversity_zero_when_uniform() {
        let d = dist(&[0.3, 0.3, 0.3]);
        assert!(macro_diversity(&d).abs() < 1e-12);
    }

    #[test]
    fn mobilized_counts_extremes() {
        let d = dist(&[0.6, -0.7, 0.1, -0.2]);
        assert_eq!(mobilized(&d, 0.5), 2);
    }

    #[test]
    fn polarization_high_for_bimodal() {
        let bimodal = dist(&[0.9, 0.9, -0.9, -0.9]);
        let consensus = dist(&[0.8, 0.8, 0.8, 0.8]);
        assert!(polarization(&bimodal) > polarization(&consensus));
        assert!((polarization(&bimodal) - 1.0).abs() < 1e-12);
        assert_eq!(polarization(&consensus), 0.0);
    }

    #[test]
    fn core_influence_averages_core_only() {
        let d = dist(&[1.0, -1.0, 0.0]);
        let mut tier = BTreeMap::new();
        tier.insert(AgentId(0), Tier::Core);
        tier.insert(AgentId(1), Tier::Ordinary);
        tier.insert(AgentId(2), Tier::Ordinary);
        assert!((core_influence(&d, &tier) - 1.0).abs() < 1e-12);
    }
}
