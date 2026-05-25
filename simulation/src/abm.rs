//! 周辺 (一般) ユーザの意見力学 ABM (論文付録 A・§6 の統一定式化)．
//!
//! Chuang & Rogers (2023) の統一定式化に従い，態度更新を 3 つの関数に分解する:
//!
//! - `f_message(a_j) = a_j` — メッセージ関数 (多くの ABM はバイアスなく態度を
//!   そのまま伝える)．[`f_message`]．
//! - `f_selection` — 相互作用相手 (可視メッセージ) の選択．[`crate::mechanisms`]
//!   側でネットワーク隣接 (+ コア層の call-to-action) から構成する．
//! - `f_update(a_i, M_i)` — 受信メッセージ集合から態度差分 Δa を計算する．
//!   本モジュールの [`AbmModel::f_update`] が担う．
//!
//! 4 モデルを実装する:
//!
//! | モデル | 規則 | 定性挙動 |
//! |--------|------|---------|
//! | BC (Deffuant) | 信頼境界内のみ同化．`Δa = α·sim·(a_j − a_i)` | 合意形成 (収束) |
//! | HK | 信頼境界内の全ソース平均へ移動 | 合意形成 (クラスタ) |
//! | SJ (Social Judgement) | 受容域は同化，拒否域は反発 (態度から遠ざかる) | 二極化 |
//! | Lorenz | 同化 + 強化 + 分極化 (極端意見を増幅) | 二極化 |
//!
//! 態度は常に `A = [-1, 1]` にクランプする．更新は **同期更新**: 呼び出し側が
//! ステップ開始時の態度スナップショットから `Δa` を計算し，一括で書き戻す
//! (mid-step の変化が同一ステップ内に波及しない)．

use crate::config::{AbmModel, AbmParams};

// 周辺層の数式は共有パック socsim-social-dynamics に委譲する (socsim #42/#43)．
// 当該 bare 関数は本リポジトリの旧 `abm.rs` から byte-for-byte 移植されたもので，
// `clamp_attitude` / `f_message` / 4 つの Δ 更新規則 (BC/HK/SJ/Lorenz) はパック側と
// 完全一致する (空メッセージ → Δ=0，信頼境界判定は strict `<`)．本リポジトリ側には
// `AbmModel` / `f_update` ディスパッチ / `AbmParams` (Mobilization メカニズムへの
// インタフェース) のみを残す．
pub use socsim_social_dynamics::updates::{ATTITUDE_MAX, ATTITUDE_MIN};
pub use socsim_social_dynamics::{clamp_attitude, f_message};

use socsim_social_dynamics::{
    bounded_confidence_update, hk_update as pack_hk_update, lorenz_update as pack_lorenz_update,
    social_judgement_update,
};

impl AbmModel {
    /// 受信メッセージ集合 `messages` (= `{ f_message(a_j) | j ∈ J_i }`) から，
    /// エージェント `a_i` の態度差分 `Δa = a_{t+1} − a_t` を計算する．
    ///
    /// 戻り値は **差分** (新態度ではない)．呼び出し側で `a_i + Δa` をクランプして
    /// 一括書き戻すこと (同期更新)．`messages` が空なら `Δa = 0`．
    ///
    /// `params` は α (同化率)・ε (信頼境界)・rejection (SJ 拒否域)・repulsion
    /// (反発/分極化強度) を持つ．
    pub fn f_update(&self, a_i: f64, messages: &[f64], params: &AbmParams) -> f64 {
        if messages.is_empty() {
            return 0.0;
        }
        match self {
            AbmModel::Bc => bc_update(a_i, messages, params),
            AbmModel::Hk => hk_update(a_i, messages, params),
            AbmModel::Sj => sj_update(a_i, messages, params),
            AbmModel::Lorenz => lorenz_update(a_i, messages, params),
        }
    }
}

/// Bounded Confidence (Deffuant et al. 2000)．
///
/// `Δa = α · sim(a_i, m_j) · (m_j − a_i)` を信頼境界内のソースについて平均する．
/// `sim = 1` (|m_j − a_i| < ε) のときのみ同化が起き，それ以外は無視．境界内ソース
/// が無ければ `Δa = 0`．→ 信頼境界内クラスタが合意形成 (収束) しやすい．
///
/// 数式は共有パックの [`bounded_confidence_update`] へ委譲する (`AbmParams` →
/// `(ε, α)` を写像)．
fn bc_update(a_i: f64, messages: &[f64], params: &AbmParams) -> f64 {
    bounded_confidence_update(a_i, messages, params.epsilon, params.alpha)
}

/// Hegselmann–Krause (2002) — 複数ソース BC．
///
/// 信頼境界 ε 内の全ソース (自身を暗黙に含む) の平均態度へ α の割合で移動する．
/// `Δa = α · (mean_{|m_j − a_i| < ε} m_j − a_i)`．境界内ソースが無ければ `Δa = 0`．
///
/// 数式は共有パックの [`socsim_social_dynamics::hk_update`] へ委譲する．
fn hk_update(a_i: f64, messages: &[f64], params: &AbmParams) -> f64 {
    pack_hk_update(a_i, messages, params.epsilon, params.alpha)
}

/// Social Judgement モデル．
///
/// 受容域 (|m_j − a_i| < ε) では同化，拒否域 (|m_j − a_i| > rejection) では反発
/// (相手の態度から遠ざかる)，その間 (非関与域) は無視．反発は二極化を生む．
///
/// 数式は共有パックの [`social_judgement_update`] へ委譲する (`AbmParams` →
/// `(ε, α, rejection, repulsion)` を写像)．
fn sj_update(a_i: f64, messages: &[f64], params: &AbmParams) -> f64 {
    social_judgement_update(
        a_i,
        messages,
        params.epsilon,
        params.alpha,
        params.rejection,
        params.repulsion,
    )
}

/// Lorenz (2021) — 同化 + 強化 + 分極化．
///
/// 受容域では同化しつつ，「強化」項で自身の現態度の符号方向へ僅かに増幅し
/// (確証バイアス的強化)，極端な意見ほど強く分極化する．境界外は無視．
/// 結果として中庸が崩れ二極化が進む．
///
/// 数式は共有パックの [`socsim_social_dynamics::lorenz_update`] へ委譲する．
fn lorenz_update(a_i: f64, messages: &[f64], params: &AbmParams) -> f64 {
    pack_lorenz_update(
        a_i,
        messages,
        params.epsilon,
        params.alpha,
        params.repulsion,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(model: AbmModel) -> AbmParams {
        AbmParams {
            model,
            alpha: 0.5,
            epsilon: 0.4,
            rejection: 0.8,
            repulsion: 0.2,
        }
    }

    #[test]
    fn empty_messages_no_change() {
        for m in [AbmModel::Bc, AbmModel::Hk, AbmModel::Sj, AbmModel::Lorenz] {
            assert_eq!(m.f_update(0.3, &[], &params(m)), 0.0);
        }
    }

    #[test]
    fn bc_assimilates_within_bound() {
        let p = params(AbmModel::Bc);
        // a_i=0.0, m_j=0.2 (within ε=0.4) → Δ = 0.5*(0.2-0) = 0.1
        let d = AbmModel::Bc.f_update(0.0, &[0.2], &p);
        assert!((d - 0.1).abs() < 1e-12);
    }

    #[test]
    fn bc_ignores_outside_bound() {
        let p = params(AbmModel::Bc);
        // a_i=0.0, m_j=0.9 (outside ε=0.4) → Δ = 0
        let d = AbmModel::Bc.f_update(0.0, &[0.9], &p);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn bc_moves_toward_neighbors() {
        // BC: 近接した意見へ収束する方向 (符号が差分と一致)．
        let p = params(AbmModel::Bc);
        let d = AbmModel::Bc.f_update(-0.1, &[0.1, 0.05], &p);
        assert!(d > 0.0, "should move toward positive neighbors");
    }

    #[test]
    fn hk_moves_toward_mean() {
        let p = params(AbmModel::Hk);
        // a_i=0.0, messages 0.2,0.2 within ε → mean of {0,0.2,0.2}=0.1333, Δ=0.5*0.1333
        let d = AbmModel::Hk.f_update(0.0, &[0.2, 0.2], &p);
        assert!(d > 0.0 && d < 0.2);
    }

    #[test]
    fn sj_repels_in_rejection_region() {
        let p = params(AbmModel::Sj);
        // a_i=0.0, m_j=0.9 (>rejection 0.8) → repel away (negative Δ)
        let d = AbmModel::Sj.f_update(0.0, &[0.9], &p);
        assert!(d < 0.0, "SJ should repel from far-positive message");
    }

    #[test]
    fn sj_assimilates_in_acceptance_region() {
        let p = params(AbmModel::Sj);
        let d = AbmModel::Sj.f_update(0.0, &[0.2], &p);
        assert!(d > 0.0, "SJ should assimilate near message");
    }

    #[test]
    fn lorenz_polarizes_extremes() {
        let p = params(AbmModel::Lorenz);
        // 極端な正の態度は，境界外メッセージのみでも分極化項で更に正へ押される．
        let d = AbmModel::Lorenz.f_update(0.8, &[-0.9], &p);
        assert!(
            d > 0.0,
            "Lorenz should push an extreme attitude further out"
        );
    }

    #[test]
    fn clamp_keeps_in_range() {
        assert_eq!(clamp_attitude(1.5), 1.0);
        assert_eq!(clamp_attitude(-2.0), -1.0);
        assert_eq!(clamp_attitude(0.3), 0.3);
    }
}
