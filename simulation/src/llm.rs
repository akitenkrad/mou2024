//! LLM harness — re-exported from socsim-llm (was a per-repo copy; consolidated).
//!
//! The «Ollama-first → OpenAI fallback + cache» builder, the `CachingClient`
//! alias, and the `LlmConfig` helper now live in `socsim-llm::harness`.  This
//! module re-exports them (preserving the repo-local `crate::llm::*` paths:
//! `HiSimClient`, `build_live_client`, `wrap_client`, `llm_config`) and keeps
//! the repo-specific [`classify_stance`] postprocessing, which is *model* logic
//! (not harness boilerplate) and stays here.
pub use socsim_llm::build_live_client_from_settings as build_live_client;
pub use socsim_llm::{llm_config, wrap_client, LiveClient as HiSimClient};

// --------------------------------------------------------------------------- //
// postprocessing: テキスト → 態度スコア (決定論的 stance/sentiment 分類器)
// --------------------------------------------------------------------------- //

/// コア LLM のテキスト応答を態度スコア $\in [-1, 1]$ へ変換する軽量・決定論的
/// 分類器 (論文の stance 注釈 + TextBlob 感情分析のスタンドイン)．
///
/// stance (態度方向) を支持/反対のキーワード差で，強度を語数の対数で近似する．
/// 同一テキストは必ず同一スコアになるため，socsim コアの再現性を壊さず，かつ
/// キャッシュと整合する．より精緻な外部 LLM 注釈はオプション (Phase 3 拡張点)．
pub fn classify_stance(text: &str) -> f64 {
    const POSITIVE: [&str; 14] = [
        "support",
        "agree",
        "favor",
        "yes",
        "right",
        "justice",
        "stand",
        "solidarity",
        "believe",
        "true",
        "good",
        "fight",
        "change",
        "must",
    ];
    const NEGATIVE: [&str; 12] = [
        "oppose",
        "disagree",
        "against",
        "no",
        "wrong",
        "false",
        "doubt",
        "fake",
        "bad",
        "stop",
        "skeptical",
        "overblown",
    ];

    let lower = text.to_ascii_lowercase();
    let mut pos = 0i32;
    let mut neg = 0i32;
    for token in lower.split(|c: char| !c.is_alphanumeric()) {
        if token.is_empty() {
            continue;
        }
        if POSITIVE.contains(&token) {
            pos += 1;
        }
        if NEGATIVE.contains(&token) {
            neg += 1;
        }
    }
    let net = pos - neg;
    if net == 0 {
        return 0.0;
    }
    // 強度は語数差の対数で飽和させ，[-1, 1] にクランプする．
    let magnitude = (1.0 + (net.unsigned_abs() as f64)).ln() / (1.0 + 5.0_f64).ln();
    (net.signum() as f64 * magnitude).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_is_deterministic() {
        let t = "I support this movement and stand in solidarity.";
        assert_eq!(classify_stance(t), classify_stance(t));
    }

    #[test]
    fn classify_positive_text() {
        let s = classify_stance("I agree and support, this is justice and we must fight.");
        assert!(s > 0.0, "support text should be positive, got {s}");
    }

    #[test]
    fn classify_negative_text() {
        let s = classify_stance("I oppose this, it is wrong and fake, I disagree.");
        assert!(s < 0.0, "opposition text should be negative, got {s}");
    }

    #[test]
    fn classify_neutral_text() {
        assert_eq!(classify_stance("the weather today is cloudy"), 0.0);
    }
}
