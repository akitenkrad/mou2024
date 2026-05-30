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
// stance annotation モード (決定論 ⇄ 外部 LLM)
// --------------------------------------------------------------------------- //

/// コア LLM が発信したメッセージ本文を態度スコア $\in [-1, 1]$ へ写像する手段．
///
/// HiSim 論文は post の stance を外部の stance 分類器 (+ TextBlob 感情分析) で
/// 注釈する．本リポジトリの既定は，それを**決定論的**に近似する軽量分類器
/// ([`classify_stance`]) である．`--stance-annotator llm` を指定すると，同じコア
/// LLM に «この post の stance を 5 段階で答えよ» と追加で問い合わせ，その応答を
/// 態度スケールへ写像する経路 ([`classify_stance_llm`]) に切り替わる (論文 §「外部
/// LLM stance 注釈」拡張点)．
///
/// 既定 [`StanceMode::Deterministic`] は従来挙動と**ビット等価** (追加の LLM 呼び出し
/// 無し)．外部 LLM 経路はキャッシュ対象なので，`temperature=0` + 固定 seed +
/// プロンプトキャッシュにより擬似決定論を保つ．
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StanceMode {
    /// 決定論的キーワード分類器 ([`classify_stance`]; 既定・追加 LLM 呼び出し無し)．
    #[default]
    Deterministic,
    /// 外部 LLM stance 注釈 ([`classify_stance_llm`]; コア LLM へ追加問い合わせ)．
    Llm,
}

impl StanceMode {
    /// 短い識別ラベル (CLI / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            StanceMode::Deterministic => "deterministic",
            StanceMode::Llm => "llm",
        }
    }
}

/// 文字列から [`StanceMode`] をパースする．
pub fn parse_stance_mode(s: &str) -> Result<StanceMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "deterministic" | "det" | "keyword" | "textblob" => Ok(StanceMode::Deterministic),
        "llm" | "external" | "annotator" => Ok(StanceMode::Llm),
        _ => Err(format!(
            "不正な stance 注釈モード: \"{}\" (deterministic / llm)",
            s
        )),
    }
}

/// 外部 LLM stance 注釈用のプロンプトを組む (5 段 Likert; 決定論的に整数を抽出)．
///
/// イベント文脈下で post の stance を «-2 (強く反対) … +2 (強く支持)» の整数で
/// 返させ，[`classify_stance_llm`] がその整数を $[-1, 1]$ へ写像する．
pub fn stance_annotation_prompt(event: &str, message: &str) -> String {
    format!(
        "You are a careful stance-annotation assistant. Given a social-media post made \
         during a social movement, classify the author's stance toward the movement on a \
         5-point scale.\n\nMovement context: {event}\nPost: \"{message}\"\n\nReply with a \
         SINGLE integer from -2 to 2, where -2 = strongly against, -1 = against, 0 = \
         neutral/unclear, 1 = in favor, 2 = strongly in favor. Answer with only the integer."
    )
}

/// 外部 LLM stance 注釈の応答 (整数 -2..2) を態度スコア $[-1, 1]$ へ写像する．
///
/// 応答テキストから最初に現れる整数 (符号付き) を抽出し，`/2.0` で $[-1, 1]$ に
/// 正規化してクランプする．整数が読めない場合は決定論的分類器 ([`classify_stance`])
/// にフォールバックする (堅牢化; 注釈 LLM が逸脱しても再現が壊れない)．
pub fn classify_stance_llm(annotation: &str, fallback_text: &str) -> f64 {
    if let Some(v) = parse_first_int(annotation) {
        return (v as f64 / 2.0).clamp(-1.0, 1.0);
    }
    classify_stance(fallback_text)
}

/// テキストから最初に現れる符号付き整数を抽出する (注釈 LLM 応答のパース)．
fn parse_first_int(text: &str) -> Option<i32> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_digit()
            || ((c == '-' || c == '+')
                && i + 1 < bytes.len()
                && (bytes[i + 1] as char).is_ascii_digit())
        {
            let start = i;
            if c == '-' || c == '+' {
                i += 1;
            }
            while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                i += 1;
            }
            return text[start..i].parse::<i32>().ok();
        }
        i += 1;
    }
    None
}

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

    #[test]
    fn stance_mode_parses_and_defaults() {
        assert_eq!(StanceMode::default(), StanceMode::Deterministic);
        assert_eq!(
            parse_stance_mode("deterministic").unwrap(),
            StanceMode::Deterministic
        );
        assert_eq!(parse_stance_mode("LLM").unwrap(), StanceMode::Llm);
        assert_eq!(parse_stance_mode("external").unwrap(), StanceMode::Llm);
        assert!(parse_stance_mode("nope").is_err());
    }

    #[test]
    fn parse_first_int_handles_signs_and_prose() {
        assert_eq!(parse_first_int("2"), Some(2));
        assert_eq!(parse_first_int("the stance is -1 here"), Some(-1));
        assert_eq!(parse_first_int("Answer: +2"), Some(2));
        assert_eq!(parse_first_int("no number"), None);
    }

    #[test]
    fn classify_stance_llm_maps_likert_to_unit() {
        assert!((classify_stance_llm("2", "ignored") - 1.0).abs() < 1e-12);
        assert!((classify_stance_llm("-2", "ignored") + 1.0).abs() < 1e-12);
        assert!((classify_stance_llm("1", "ignored") - 0.5).abs() < 1e-12);
        assert_eq!(classify_stance_llm("0", "ignored"), 0.0);
    }

    #[test]
    fn classify_stance_llm_falls_back_when_unparseable() {
        // 注釈 LLM が整数を返さないとき → 決定論的分類器に倒す (再現を壊さない)．
        let fallback = "I support this and stand in solidarity for justice.";
        assert_eq!(
            classify_stance_llm("I cannot decide", fallback),
            classify_stance(fallback)
        );
    }
}
