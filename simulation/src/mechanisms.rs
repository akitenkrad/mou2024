//! socsim フレームワーク上の HiSim メカニズム (4 機構 × フェーズ)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! 外部イベント提示・周辺 ABM 更新・マクロ集約をグラフ構造と `ctx.rng` で行い，
//! 上層 (非決定的 LLM レイヤ) は [`DecisionMechanism`] の `Decision` フェーズの
//! **コア層処理にのみ** 閉じ込める．
//!
//! # Mechanism × Phase
//!
//! | Mechanism | Phase | 役割 |
//! |-----------|-------|------|
//! | [`EnvironmentMechanism`]   | Environment | トリガーニュース・外部イベントを当該ステップに提示 |
//! | [`DecisionMechanism`]      | Decision    | **コア層**: LLM で行動生成 → stance 分類で態度更新．**周辺層**: ABM f_selection で相互作用相手を決定 |
//! | [`MobilizationMechanism`]  | Interaction | コア→周辺の意見伝播．周辺層は f_message→f_update で態度を同期更新 (一方向結合) |
//! | [`AggregateMechanism`]     | Reward      | マクロ集約 (Bias/Diversity/動員/分極化) を計算・記録．T 到達/収束で request_stop |

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::abm::{clamp_attitude, f_message};
use crate::config::{AbmParams, LlmSettings};
use crate::llm::{classify_stance, llm_config, HiSimClient};
use crate::parse::{parse_action, ActionDecision};
use crate::prompts::action_prompt;
use crate::world::{HiSimWorld, Tier};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<HiSimClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;
/// 共有 LLM 呼び出し予算カウンタ (run 全体で残数を管理)．
pub type SharedBudget = Rc<RefCell<usize>>;

/// scratch キー: 当該ステップでコア層が発信したメッセージ (態度値; Decision → Interaction)．
const SCRATCH_BROADCASTS: &str = "core_broadcasts";
/// scratch キー: 当該ステップで LLM を呼んだコアエージェント数．
const SCRATCH_LLM_ACTIONS: &str = "llm_actions";

// --------------------------------------------------------------------------- //
// EnvironmentMechanism (Environment)
// --------------------------------------------------------------------------- //

/// 外部イベント提示 (`Environment`)．
///
/// 当該ステップに対応するトリガーニュース・外部イベントを `world.current_event`
/// に設定する．データセット名に応じたイベントスケジュール (single / phased) を持つ
/// 決定論的処理 (LLM 非依存)．コア層プロンプトの文脈に使う．
pub struct EnvironmentMechanism {
    dataset: String,
}

impl EnvironmentMechanism {
    /// データセット名から作る．
    pub fn new(dataset: &str) -> Self {
        EnvironmentMechanism {
            dataset: dataset.to_string(),
        }
    }

    /// データセット・ステップに対応するイベント文を返す (決定論的)．
    fn event_for(&self, t: u64) -> String {
        let topic = match self.dataset.to_ascii_lowercase().as_str() {
            "metoo" => "the #MeToo movement",
            "roe" | "roeoverturned" => "the overturning of Roe v. Wade",
            "blm" | "blacklivesmatter" => "the Black Lives Matter movement",
            other => return format!("step {t}: ongoing discussion about {other}"),
        };
        match t {
            0 => format!("A trigger event ignites public discussion about {topic}."),
            1..=3 => format!("News coverage and viral testimonies amplify {topic}."),
            4..=7 => format!("Counter-narratives and debate spread around {topic}."),
            _ => format!("Sustained mobilization and reflection continue around {topic}."),
        }
    }
}

impl Mechanism<HiSimWorld> for EnvironmentMechanism {
    fn name(&self) -> &str {
        "environment"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, HiSimWorld>) -> Result<()> {
        let event = self.event_for(ctx.clock.t());
        ctx.world.current_event = event;
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// DecisionMechanism (Decision) — LLM レイヤ境界
// --------------------------------------------------------------------------- //

/// 行動決定 (`Decision`) — **二層境界の唯一の LLM 呼び出し点**．
///
/// - **コア層**: profile/memory + 外部イベント + 隣接態度を文脈に LLM を呼び，
///   行動 (post/retweet/reply/like/do-nothing) を生成する．発信行動
///   (post/retweet/reply) なら応答メッセージを stance 分類して自身の態度を更新し，
///   そのメッセージ態度を scratch (`SCRATCH_BROADCASTS`) へ書く (周辺層への影響源)．
/// - **周辺層**: ここでは何もしない (相互作用相手の選択 = f_selection は
///   `MobilizationMechanism` がネットワーク隣接から行う)．
///
/// LLM 予算 (`SharedBudget`) を超えたコアは do-nothing に倒す (キャリブレーション
/// 用のコスト制御)．`core_ratio = 0.0` ならコアが 0 体なので LLM は一切呼ばれない．
pub struct DecisionMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    budget: SharedBudget,
    settings: LlmSettings,
}

impl DecisionMechanism {
    /// 共有クライアント・メタデータ・予算・LLM 設定から作る．
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        budget: SharedBudget,
        settings: LlmSettings,
    ) -> Self {
        DecisionMechanism {
            client,
            metadata,
            budget,
            settings,
        }
    }
}

impl Mechanism<HiSimWorld> for DecisionMechanism {
    fn name(&self) -> &str {
        "decision"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, HiSimWorld>) -> Result<()> {
        // 当該ステップにコア層が発信したメッセージ態度 (周辺層が次フェーズで受信)．
        let mut broadcasts: BTreeMap<AgentId, f64> = BTreeMap::new();
        let mut llm_actions = 0usize;

        // コア層は AgentId 昇順で処理する (BTreeMap キーはソート済み; 決定論)．
        let core_ids: Vec<AgentId> = ctx
            .world
            .tier
            .iter()
            .filter(|(_, t)| matches!(t, Tier::Core))
            .map(|(&id, _)| id)
            .collect();

        let event = ctx.world.current_event.clone();

        for id in core_ids {
            if *self.budget.borrow() == 0 {
                break;
            }
            let attitude = *ctx.world.attitude.get(&id).unwrap_or(&0.0);
            let (profile, memory) = match ctx.world.core.get(&id) {
                Some(cs) => (cs.profile.clone(), cs.memory.clone()),
                None => continue,
            };
            let neighbor_attitudes: Vec<f64> = ctx
                .world
                .network
                .neighbors(id)
                .into_iter()
                .filter_map(|nb| ctx.world.attitude.get(&nb).copied())
                .collect();

            let prompt = action_prompt(&profile, &memory, &event, attitude, &neighbor_attitudes);
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client
                    .complete(&prompt, &llm_config(&self.settings))
                    .map_err(|e| {
                        SocsimError::Mechanism(format!("core decision LLM call failed: {e}"))
                    })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            {
                let mut b = self.budget.borrow_mut();
                *b = b.saturating_sub(1);
            }
            llm_actions += 1;

            let decision: ActionDecision = parse_action(&text);

            // 発信行動なら stance 分類でメッセージ態度を得て自身を更新 + broadcast．
            if decision.kind.broadcasts() {
                let msg = decision.message.clone().unwrap_or_default();
                let stance = classify_stance(&msg);
                // コア自身の態度をメッセージ stance 方向へ僅かに更新 (postprocessing)．
                let new_attitude = clamp_attitude(attitude + 0.3 * (stance - attitude));
                if let Some(a) = ctx.world.attitude.get_mut(&id) {
                    *a = new_attitude;
                }
                // 記憶へイベント・自分の行動を積む (上限 10)．
                if let Some(cs) = ctx.world.core.get_mut(&id) {
                    cs.memory
                        .push(format!("[{}] {}", decision.kind.label(), msg));
                    if cs.memory.len() > 10 {
                        let overflow = cs.memory.len() - 10;
                        cs.memory.drain(0..overflow);
                    }
                }
                // 発信メッセージ態度 (周辺層への影響源)．
                broadcasts.insert(id, new_attitude);
            } else if let Some(cs) = ctx.world.core.get_mut(&id) {
                // like / do-nothing: 記憶のみ更新 (態度据え置き)．
                cs.memory.push(format!("[{}]", decision.kind.label()));
                if cs.memory.len() > 10 {
                    let overflow = cs.memory.len() - 10;
                    cs.memory.drain(0..overflow);
                }
            }
        }

        ctx.scratch.insert(SCRATCH_BROADCASTS, broadcasts);
        ctx.scratch.insert(SCRATCH_LLM_ACTIONS, llm_actions);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// MobilizationMechanism (Interaction)
// --------------------------------------------------------------------------- //

/// 動員・意見伝播 (`Interaction`)．
///
/// 周辺 (Ordinary) 層の態度を ABM で **同期更新** する:
///
/// 1. ステップ開始時の態度をスナップショット `prev` (同期更新の正本)．
/// 2. 各周辺エージェント i について受信メッセージ集合 `M_i` を構成する
///    (= f_selection): ネットワーク隣接ノードの `f_message(a_j) = a_j` を集め，
///    さらにコア層が当該ステップに発信したメッセージ態度 (`broadcasts`) のうち i の
///    フォロー先のものを加える (コア→周辺の call-to-action; **一方向結合**)．
/// 3. `abm.f_update(a_i, M_i)` で態度差分 Δa を計算する．
/// 4. 全周辺エージェントの新態度を一括書き戻す (同期更新)．
///
/// コア層の態度は周辺の影響を受けない (周辺→コアは微小として無視; 一方向結合)．
pub struct MobilizationMechanism {
    abm: AbmParams,
}

impl MobilizationMechanism {
    /// ABM パラメータから作る．
    pub fn new(abm: AbmParams) -> Self {
        MobilizationMechanism { abm }
    }
}

impl Mechanism<HiSimWorld> for MobilizationMechanism {
    fn name(&self) -> &str {
        "mobilization"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, HiSimWorld>) -> Result<()> {
        // ステップ開始時の態度スナップショット (同期更新の正本)．
        let prev: BTreeMap<AgentId, f64> = ctx.world.attitude.clone();

        // コア層が当該ステップに発信したメッセージ態度 (Decision フェーズが書いた)．
        let broadcasts: BTreeMap<AgentId, f64> = ctx
            .scratch
            .get::<BTreeMap<AgentId, f64>>(SCRATCH_BROADCASTS)
            .cloned()
            .unwrap_or_default();

        // 周辺層のみ更新する (AgentId 昇順; 決定論)．
        let ordinary_ids: Vec<AgentId> = ctx
            .world
            .tier
            .iter()
            .filter(|(_, t)| matches!(t, Tier::Ordinary))
            .map(|(&id, _)| id)
            .collect();

        let mut new_attitudes: Vec<(AgentId, f64)> = Vec::with_capacity(ordinary_ids.len());

        for id in ordinary_ids {
            let a_i = *prev.get(&id).unwrap_or(&0.0);

            // f_selection: ネットワーク隣接ノードの (旧) 態度をメッセージとして集める．
            let mut messages: Vec<f64> = Vec::new();
            for nb in ctx.world.network.neighbors(id) {
                if let Some(&a_j) = prev.get(&nb) {
                    messages.push(f_message(a_j));
                    // コアが当該ステップに発信していれば call-to-action を加算 (一方向結合)．
                    if let Some(&bc) = broadcasts.get(&nb) {
                        messages.push(f_message(bc));
                    }
                }
            }

            let delta = self.abm.model.f_update(a_i, &messages, &self.abm);
            new_attitudes.push((id, clamp_attitude(a_i + delta)));
        }

        // 一括書き戻し (同期更新)．
        for (id, a) in new_attitudes {
            ctx.world.attitude.insert(id, a);
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// AggregateMechanism (Reward)
// --------------------------------------------------------------------------- //

/// マクロ集約 (`Reward`)．
///
/// 当該ステップの態度分布から macro_bias / macro_diversity / mobilized /
/// polarization / core_influence を計算し，`world` のマクロフィールドと recorder へ
/// 記録する．収束 (態度変化が十分小さい) を検知したら `request_stop` する
/// (T 到達はエンジンのクロックが担うため明示停止は収束時のみ)．
pub struct AggregateMechanism {
    /// 動員判定の態度しきい値．
    threshold: f64,
    /// 収束判定の許容誤差 (前ステップとの bias 差がこれ未満なら収束)．
    tol: f64,
    /// 前ステップの macro_bias (収束判定用)．
    prev_bias: RefCell<Option<f64>>,
}

impl AggregateMechanism {
    /// 動員しきい値と収束許容誤差から作る．
    pub fn new(threshold: f64, tol: f64) -> Self {
        AggregateMechanism {
            threshold,
            tol,
            prev_bias: RefCell::new(None),
        }
    }
}

impl Mechanism<HiSimWorld> for AggregateMechanism {
    fn name(&self) -> &str {
        "aggregate"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, HiSimWorld>) -> Result<()> {
        let bias = crate::metrics::macro_bias(&ctx.world.attitude);
        let diversity = crate::metrics::macro_diversity(&ctx.world.attitude);
        let mob = crate::metrics::mobilized(&ctx.world.attitude, self.threshold);
        let pol = crate::metrics::polarization(&ctx.world.attitude);

        ctx.world.macro_bias = bias;
        ctx.world.macro_diversity = diversity;
        ctx.world.mobilized = mob;

        let t = ctx.clock.t();
        ctx.recorder.record_metric(t, "macro_bias", bias);
        ctx.recorder.record_metric(t, "macro_diversity", diversity);
        ctx.recorder.record_metric(t, "mobilized", mob as f64);
        ctx.recorder.record_metric(t, "polarization", pol);

        // 収束判定: 前ステップ bias との差が tol 未満なら停止 (純粋 ABM の不動点)．
        let converged = match *self.prev_bias.borrow() {
            Some(prev) => (bias - prev).abs() < self.tol,
            None => false,
        };
        *self.prev_bias.borrow_mut() = Some(bias);
        if converged {
            ctx.request_stop();
        }
        Ok(())
    }
}
