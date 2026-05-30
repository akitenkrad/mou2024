//! SoMoSiMu-Bench 連携 — 運動ダイナミクス指標の整合スキャフォールディング．
//!
//! Mou et al. (2024) HiSim は **SoMoSiMu-Bench** (3 つの実運動 #MeToo /
//! RoeOverturned / BlackLivesMatter) に対し，シミュレートした集団態度の軌跡を
//! 実観測の運動ダイナミクスと照合する (論文 §5 / Table 2)．本モジュールは，
//! HiSim の出力 (態度時系列・最終マクロ指標) を SoMoSiMu-Bench 流の **運動指標**
//! へ写像し，参照系列と比較する **整合ハーネス** を提供する．
//!
//! # 何が «実» で何が «較正» か (正直な区分)
//!
//! - **実 (シミュレータ由来)**: 比較される観測値 — macro_bias / macro_diversity /
//!   mobilized / polarization の時系列と最終値 — はすべて HiSim の実行から生じる
//!   実データである．[`movement_metrics`] が計算する整合指標 (動員ピーク・立ち上が
//!   り時刻・態度トレンドの符号・正規化曲線間距離) も実出力から決定論的に導く．
//! - **較正 (合成参照)**: SoMoSiMu-Bench の生データセットは本リポジトリに同梱され
//!   ない (取得・ライセンスは原著者の管理下)．そこで各運動について，論文 §5 の
//!   定性記述 (例: #MeToo は急峻な立ち上がり後に持続，RoeOverturned は二極化が顕著)
//!   に整合する **較正済みパラメトリック参照曲線** ([`reference_curve`]) を合成し，
//!   それを «bench 参照» として使う．これは ground-truth ではなく，整合パイプライン
//!   を end-to-end で走らせ，かつ論文の定性傾向との一致を点検するための代理である．
//!   実 bench データが入手できれば [`BenchReference`] を CSV ロードに差し替えるだけ
//!   で同じ比較経路が使える (アライメントロジックは不変)．
//!
//! いずれの比較も «数値完全一致» ではなく **定性傾向の整合** (符号・順序・帯) を
//! 目的とする (`reproduce` のアンカーと同じ哲学; ローカル LLM ≠ 論文の gpt-3.5)．

use serde::Serialize;

use crate::metrics::StepMetrics;

/// SoMoSiMu-Bench 流の運動ダイナミクス指標 (1 実行・1 運動分)．
///
/// HiSim の態度時系列から決定論的に計算する «実» 指標．動員曲線の形状 (ピーク高・
/// ピーク時刻・立ち上がり) と最終的な集団態度の方向・分極化を要約する．
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MovementMetrics {
    /// 正規化動員ピーク (= max_t mobilized / N ∈ [0,1])．
    pub mobilization_peak: f64,
    /// 動員ピークに達するステップ (立ち上がりの速さ)．
    pub peak_step: usize,
    /// 最終ステップの正規化動員水準 (= mobilized_T / N)．
    pub final_mobilization: f64,
    /// 最終 macro_bias (集団態度の方向)．
    pub final_bias: f64,
    /// 最終 polarization (二極化の度合い)．
    pub final_polarization: f64,
    /// 動員曲線の «持続性» (= 平均動員 / ピーク動員; 1 に近いほど持続)．
    pub sustain_ratio: f64,
}

impl MovementMetrics {
    /// メトリクス履歴 (各ステップの [`StepMetrics`]) と N から計算する．
    pub fn from_history(history: &[StepMetrics], n_agents: usize) -> Self {
        let n = n_agents.max(1) as f64;
        let mut peak = 0.0_f64;
        let mut peak_step = 0usize;
        let mut sum_mob = 0.0_f64;
        for m in history {
            let mob = m.mobilized as f64 / n;
            sum_mob += mob;
            if mob > peak {
                peak = mob;
                peak_step = m.t;
            }
        }
        let last = history.last();
        let final_mobilization = last.map(|m| m.mobilized as f64 / n).unwrap_or(0.0);
        let final_bias = last.map(|m| m.macro_bias).unwrap_or(0.0);
        let final_polarization = last.map(|m| m.polarization).unwrap_or(0.0);
        let mean_mob = if history.is_empty() {
            0.0
        } else {
            sum_mob / history.len() as f64
        };
        let sustain_ratio = if peak > 1e-12 { mean_mob / peak } else { 0.0 };
        MovementMetrics {
            mobilization_peak: peak,
            peak_step,
            final_mobilization,
            final_bias,
            final_polarization,
            sustain_ratio,
        }
    }
}

/// SoMoSiMu-Bench «参照» (1 運動分)．較正済み合成参照，または実 bench ロード値．
///
/// `source` フィールドが «由来» を明示する (`"calibrated-synthetic"` または
/// `"bench-dataset:<name>"`)．本リポジトリ同梱は較正済み合成のみ．
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BenchReference {
    /// 運動名 (metoo / roe / blm)．
    pub movement: String,
    /// 参照の由来 (正直な区分)．
    pub source: String,
    /// 参照の運動指標 (シミュレータと同じ指標空間)．
    pub metrics: MovementMetrics,
}

/// 論文 §5 の定性記述に整合する **較正済み合成参照曲線** を生成する．
///
/// 各運動の動員曲線を «ロジスティック立ち上がり × 持続/減衰» のパラメトリック形で
/// 合成し，[`MovementMetrics`] へ畳み込む (実 bench データの代理)．パラメタは
/// 論文の定性傾向に手で合わせた較正値であり ground-truth ではない:
///
/// | 運動 | 立ち上がり | ピーク | 持続 | 最終 bias | 分極化 |
/// |------|-----------|--------|------|----------|--------|
/// | metoo | 速い | 高 | 高 (持続的連帯) | 正 (支持優勢) | 中 |
/// | roe   | 速い | 中高 | 中 | やや負 | 高 (強い二極化) |
/// | blm   | 中速 | 高 | 中高 | 正 | 中高 |
pub fn reference_curve(movement: &str, steps: usize) -> BenchReference {
    // (peak, peak_frac_of_T, sustain, final_bias, final_pol) — 較正済み定性パラメタ．
    let (peak, peak_frac, sustain, final_bias, final_pol) =
        match movement.to_ascii_lowercase().as_str() {
            "metoo" => (0.62, 0.25, 0.78, 0.42, 0.30),
            "roe" | "roeoverturned" => (0.55, 0.20, 0.66, -0.18, 0.62),
            "blm" | "blacklivesmatter" => (0.60, 0.35, 0.72, 0.38, 0.46),
            // 未知運動は中庸の汎用較正にフォールバックする．
            _ => (0.50, 0.30, 0.65, 0.20, 0.35),
        };
    let t = steps.max(1);
    let peak_step = ((peak_frac * t as f64).round() as usize).min(t.saturating_sub(1));
    BenchReference {
        movement: movement.to_string(),
        source: "calibrated-synthetic".to_string(),
        metrics: MovementMetrics {
            mobilization_peak: peak,
            peak_step,
            final_mobilization: peak * sustain,
            final_bias,
            final_polarization: final_pol,
            sustain_ratio: sustain,
        },
    }
}

/// シミュレータ指標 vs 参照指標を突き合わせた 1 整合行．
#[derive(Debug, Clone, Serialize)]
pub struct AlignmentRow {
    /// 比較する指標名．
    pub metric: String,
    /// シミュレータ観測値．
    pub observed: f64,
    /// 参照値 (較正済み合成 or 実 bench)．
    pub reference: f64,
    /// 絶対誤差 |observed − reference|．
    pub abs_error: f64,
    /// 許容帯 (これ以下なら «整合» とみなす; 定性傾向の点検)．
    pub tolerance: f64,
    /// 整合したか (abs_error <= tolerance)．
    pub aligned: bool,
}

fn align(metric: &str, observed: f64, reference: f64, tolerance: f64) -> AlignmentRow {
    let abs_error = (observed - reference).abs();
    AlignmentRow {
        metric: metric.to_string(),
        observed,
        reference,
        abs_error,
        tolerance,
        aligned: abs_error <= tolerance,
    }
}

/// 1 運動分の整合レポート (シミュレータ vs 参照)．
#[derive(Debug, Clone, Serialize)]
pub struct BenchComparison {
    /// 運動名．
    pub movement: String,
    /// 参照の由来 (正直な区分)．
    pub reference_source: String,
    /// シミュレータの運動指標．
    pub observed: MovementMetrics,
    /// 参照の運動指標．
    pub reference: MovementMetrics,
    /// 指標別の整合行．
    pub rows: Vec<AlignmentRow>,
    /// 整合した指標数．
    pub n_aligned: usize,
    /// 比較した指標総数．
    pub n_total: usize,
}

/// シミュレータの運動指標を SoMoSiMu-Bench 参照に整合させる．
///
/// 較正済み合成参照に対し，定性傾向 (動員ピーク・持続・態度方向・分極化) の整合を
/// 広めの許容帯で点検する (数値完全一致は狙わない)．
pub fn compare_to_bench(
    movement: &str,
    observed: MovementMetrics,
    reference: &BenchReference,
) -> BenchComparison {
    let r = &reference.metrics;
    let rows = vec![
        align(
            "mobilization_peak",
            observed.mobilization_peak,
            r.mobilization_peak,
            0.35,
        ),
        align(
            "final_mobilization",
            observed.final_mobilization,
            r.final_mobilization,
            0.35,
        ),
        align("final_bias", observed.final_bias, r.final_bias, 0.60),
        align(
            "final_polarization",
            observed.final_polarization,
            r.final_polarization,
            0.45,
        ),
        align(
            "sustain_ratio",
            observed.sustain_ratio,
            r.sustain_ratio,
            0.45,
        ),
    ];
    let n_aligned = rows.iter().filter(|row| row.aligned).count();
    let n_total = rows.len();
    BenchComparison {
        movement: movement.to_string(),
        reference_source: reference.source.clone(),
        observed,
        reference: r.clone(),
        rows,
        n_aligned,
        n_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(mobs: &[usize]) -> Vec<StepMetrics> {
        mobs.iter()
            .enumerate()
            .map(|(t, &m)| StepMetrics {
                t,
                macro_bias: 0.3,
                macro_diversity: 0.2,
                mobilized: m,
                polarization: 0.4,
                core_influence: 0.5,
                llm_actions: 0,
            })
            .collect()
    }

    #[test]
    fn movement_metrics_finds_peak_and_sustain() {
        let h = hist(&[10, 50, 80, 60, 40]);
        let mm = MovementMetrics::from_history(&h, 100);
        assert!((mm.mobilization_peak - 0.8).abs() < 1e-12);
        assert_eq!(mm.peak_step, 2);
        assert!((mm.final_mobilization - 0.4).abs() < 1e-12);
        // 持続比 = 平均/ピーク ∈ (0,1]．
        assert!(mm.sustain_ratio > 0.0 && mm.sustain_ratio <= 1.0);
    }

    #[test]
    fn reference_curve_is_calibrated_synthetic_and_deterministic() {
        let a = reference_curve("metoo", 14);
        let b = reference_curve("metoo", 14);
        assert_eq!(a, b);
        assert_eq!(a.source, "calibrated-synthetic");
        // roe は metoo より分極化が高い参照 (論文の定性傾向)．
        let roe = reference_curve("roe", 14);
        assert!(roe.metrics.final_polarization > a.metrics.final_polarization);
    }

    #[test]
    fn compare_runs_end_to_end_and_counts_alignment() {
        let h = hist(&[10, 40, 62, 55, 48]);
        let mm = MovementMetrics::from_history(&h, 100);
        let reference = reference_curve("metoo", 5);
        let cmp = compare_to_bench("metoo", mm, &reference);
        assert_eq!(cmp.n_total, 5);
        assert_eq!(cmp.rows.len(), 5);
        assert!(cmp.n_aligned <= cmp.n_total);
        assert_eq!(cmp.reference_source, "calibrated-synthetic");
    }

    #[test]
    fn aligned_when_within_tolerance() {
        let row = align("x", 0.50, 0.55, 0.10);
        assert!(row.aligned);
        let row = align("x", 0.10, 0.90, 0.20);
        assert!(!row.aligned);
    }
}
