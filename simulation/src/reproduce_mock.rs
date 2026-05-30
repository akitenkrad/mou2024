//! オフライン (ライブ LLM 不要) 再現用のスクリプト化クライアント．
//!
//! `reproduce --mock` と各種テストが共用する決定論的 mock を提供する．コアエージェ
//! ント (LLM 駆動) は «支持を表明し連帯を呼びかける post» を返す ScriptedClient で
//! 駆動する．これにより，**ハイブリッド経路** (LLM コア + ABM 周辺) をライブ LLM
//! 無しで end-to-end に走らせ，純 ABM (`--core-ratio 0`) と対比できる．
//!
//! mock は ground-truth LLM ではなく，論文の定性挙動 (コアが call-to-action を発信し
//! 周辺の動員を牽引する) を再現するための **戯画** である．サンドボックス・CI では
//! `--mock` を付けてライブ Ollama/OpenAI を回避する (純 ABM 経路は mock 不要で
//! そもそも LLM を呼ばない)．

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

use crate::llm::{wrap_client, HiSimClient};

/// stance 注釈プロンプトを判別するためのマーカ ([`crate::llm::stance_annotation_prompt`]
/// と一致させる)．注釈プロンプトには整数 (5 段 Likert) を，行動プロンプトには
/// post 行動を返す．
const STANCE_ANNOTATION_MARK: &str = "classify the author's stance";

/// 再現用の決定論的 scripted クライアントを構築する (in-memory cache)．
///
/// - 行動プロンプト → 支持 post (call-to-action)．
/// - stance 注釈プロンプト ([`crate::llm::StanceMode::Llm`] 経路) → 整数 `2`
///   (= 強い支持; 態度スケールで +1.0)．
pub fn build_reproduce_client() -> HiSimClient {
    let backend = ScriptedClient::new("mock-reproduce", |prompt: &str| {
        if prompt.contains(STANCE_ANNOTATION_MARK) {
            "2".to_string()
        } else {
            "THOUGHT: I will speak up.\nACTION: post\nMESSAGE: I support this movement and we must \
             stand in solidarity for justice."
                .to_string()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

#[cfg(test)]
mod tests {
    use super::*;
    use socsim_llm::LlmConfig;

    #[test]
    fn returns_support_post_for_action_prompt() {
        let mut c = build_reproduce_client();
        let r = c
            .complete(
                "Decide your next single action.",
                &LlmConfig::deterministic(),
            )
            .unwrap();
        assert!(r.text.contains("ACTION: post"));
    }

    #[test]
    fn returns_integer_for_stance_annotation_prompt() {
        let mut c = build_reproduce_client();
        let prompt = "...classify the author's stance toward the movement...";
        let r = c.complete(prompt, &LlmConfig::deterministic()).unwrap();
        assert_eq!(r.text.trim(), "2");
    }
}
