//! socsim フレームワーク上の HiSim (Mou et al. 2024) の世界状態．
//!
//! エージェントは移動する空間主体ではなく，**社会ネットワーク上のノード** であり，
//! 2 階層 (コア = LLM 駆動，周辺 = ABM 駆動) に分かれる．両者を 1 つの社会
//! ネットワーク上に配置し，態度スコア $a_{i,t} \in [-1, 1]$ を全エージェント共通の
//! 連続値で持つ．コアエージェントのみ LLM 用の profile/memory を追加で保持する．
//!
//! 空間プリミティブ (`socsim-grid`) は使わず，網プリミティブ
//! [`socsim_net::SocialNetwork`] を採用する．生成器は既定で Barabási–Albert
//! (スケールフリー = Pareto 分布と整合)．
//!
//! `agent_ids()` は `BTreeMap` のソート済みキーを返し決定論を保証する (socsim
//! コア層)．

use std::collections::BTreeMap;

use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::SocialNetwork;

/// エージェント階層 (コア = LLM 駆動，周辺 = ABM 駆動)．
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// コア層 (オピニオンリーダー等; LLM で行動を生成する)．
    Core,
    /// 周辺層 (沈黙する多数派; 決定論的 ABM で態度更新する)．
    Ordinary,
}

impl Tier {
    /// 短いラベル．
    pub fn label(&self) -> &'static str {
        match self {
            Tier::Core => "core",
            Tier::Ordinary => "ordinary",
        }
    }
}

/// コア層の重い状態 (LLM 駆動)．profile/memory はプロンプト生成に使う．
#[derive(Clone, Debug)]
pub struct CoreState {
    /// 人口統計 + 社会的特性 + コミュニケーション上の役割の自然文記述．
    pub profile: String,
    /// 個人的経験 + イベント記憶 (プロンプト文脈の元)．
    pub memory: Vec<String>,
}

impl CoreState {
    /// プロフィール文字列から空メモリのコア状態を作る．
    pub fn new(profile: String) -> Self {
        CoreState {
            profile,
            memory: Vec::new(),
        }
    }
}

/// HiSim の世界状態 (2 階層 + 社会ネットワーク)．
///
/// `#[derive(Clone)]` でスナップショット (save/resume) と比較実験に対応する．
#[derive(Clone)]
pub struct HiSimWorld {
    /// シミュレーションクロック．
    pub clock: SimClock,
    /// 全エージェント共通: 態度スコア $a_{i,t} \in [-1, 1]$ (ソート順保証)．
    pub attitude: BTreeMap<AgentId, f64>,
    /// 階層ラベル (Core / Ordinary)．
    pub tier: BTreeMap<AgentId, Tier>,
    /// コア層のみの重い状態 (LLM 用 profile/memory)．
    pub core: BTreeMap<AgentId, CoreState>,
    /// 実フォロー関係を模した社会ネットワーク (可視エージェント集合の正本)．
    pub network: SocialNetwork,
    /// 当該ステップで提示されている外部イベント (Environment が更新)．
    pub current_event: String,
    /// マクロ集約: 全エージェント態度の平均 (= 集団態度の偏り)．
    pub macro_bias: f64,
    /// マクロ集約: 態度分布の分散 (意見多様性)．
    pub macro_diversity: f64,
    /// マクロ集約: 動員された (|態度| >= 閾値) エージェント数．
    pub mobilized: usize,
}

impl HiSimWorld {
    /// 構成済みフィールドから世界状態を組み立てる (網生成・初期化は
    /// [`crate::simulation::init_world`])．
    pub fn new(
        network: SocialNetwork,
        attitude: BTreeMap<AgentId, f64>,
        tier: BTreeMap<AgentId, Tier>,
        core: BTreeMap<AgentId, CoreState>,
        steps: u64,
    ) -> Self {
        HiSimWorld {
            clock: SimClock::new(steps),
            attitude,
            tier,
            core,
            network,
            current_event: String::new(),
            macro_bias: 0.0,
            macro_diversity: 0.0,
            mobilized: 0,
        }
    }

    /// エージェント数 N．
    pub fn n(&self) -> usize {
        self.attitude.len()
    }

    /// コア層の数．
    pub fn n_core(&self) -> usize {
        self.tier.values().filter(|t| **t == Tier::Core).count()
    }

    /// 集団の平均態度 ā (= macro_bias の元)．
    pub fn mean_attitude(&self) -> f64 {
        if self.attitude.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.attitude.values().sum();
        sum / self.attitude.len() as f64
    }

    /// あるエージェントが Core 層か．
    pub fn is_core(&self, id: AgentId) -> bool {
        matches!(self.tier.get(&id), Some(Tier::Core))
    }
}

impl WorldState for HiSimWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap のキーはソート済み．契約 (sorted) を明示する．
        self.attitude.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_world() -> HiSimWorld {
        let ids: Vec<AgentId> = (0..4u64).map(AgentId).collect();
        let mut attitude = BTreeMap::new();
        let mut tier = BTreeMap::new();
        let core = BTreeMap::new();
        for (i, &id) in ids.iter().enumerate() {
            attitude.insert(id, if i % 2 == 0 { 0.5 } else { -0.5 });
            tier.insert(id, Tier::Ordinary);
        }
        let mut rng = socsim_core::SimRng::from_seed(0);
        let net = SocialNetwork::barabasi_albert(&ids, 2, &mut rng);
        HiSimWorld::new(net, attitude, tier, core, 14)
    }

    #[test]
    fn mean_attitude_is_average() {
        let w = tiny_world();
        assert!((w.mean_attitude() - 0.0).abs() < 1e-12);
        assert_eq!(w.n(), 4);
    }

    #[test]
    fn agent_ids_are_sorted() {
        let w = tiny_world();
        let ids = w.agent_ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }
}
