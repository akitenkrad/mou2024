//! LLM クライアント層 (Ollama 第一 → OpenAI フォールバック + キャッシュ)．
//!
//! 本モジュールは `socsim-llm` の合成 API に対する薄いビルダである．二層
//! アーキテクチャの **上層 (非決定的 LLM レイヤ)** をここに閉じ込め，下層の
//! 決定論的 socsim コアからは [`HiSimClient`] 型エイリアス経由でのみ触れる．
//!
//! # 合成 (Ollama 第一 → OpenAI フォールバック → キャッシュ)
//!
//! ```text
//! CachingClient< FallbackClient< OllamaClient, OpenAiClient > >
//!   └─ cache: PromptCache (prompt → response; 擬似決定論の本体)
//!      └─ primary:   OllamaClient   (OLLAMA_HOST / OLLAMA_MODEL)
//!         secondary: OpenAiClient   (OPENAI_API_KEY / OPENAI_MODEL)
//! ```
//!
//! `FallbackClient` は socsim-llm が提供する (自前実装しない)．「Ollama を試行
//! → 任意のエラーで OpenAI へフォールバック」を担う．`CachingClient` はその上に
//! プロンプト→応答キャッシュを被せ，`temperature=0` / `seed` 固定と合わせて
//! 再実行を擬似決定論化する．
//!
//! テストでは `socsim-llm::mock::ScriptedClient` を `Box<dyn LlmClient>` として
//! 同じ [`HiSimClient`] に流し込める (`impl LlmClient for Box<T>`; issue #26)．

use std::path::Path;

use socsim_llm::{CachingClient, LlmClient, LlmConfig, LlmError, PromptCache};

use crate::config::LlmSettings;

/// 本シミュレーションが用いるキャッシュ付きクライアント型．
///
/// バックエンドは `Box<dyn LlmClient>` に型消去してあり，本番は
/// `FallbackClient<OllamaClient, OpenAiClient>`，テストは `ScriptedClient` を
/// 注入できる．`socsim-llm` の `impl LlmClient for Box<T>` (issue #26) により
/// 専用 newtype なしで `CachingClient` の `C: LlmClient` 境界を満たす．
pub type HiSimClient = CachingClient<Box<dyn LlmClient>>;

/// 本番用の «Ollama 第一 → OpenAI フォールバック + キャッシュ» クライアントを
/// 環境変数から構築する．
///
/// - Ollama: `OLLAMA_HOST` (既定 `http://localhost:11434`) / `OLLAMA_MODEL`
///   (既定 `llama3.2:latest`)．
/// - OpenAI: `OPENAI_API_KEY` / `OPENAI_MODEL` (既定 `gpt-4o-mini`)．未設定なら
///   空キーのフォールバックを置く (Ollama が成功すれば呼ばれない)．
/// - キャッシュ: `settings.cache_path` があればその JSON ファイル，なければ
///   in-memory．
pub fn build_live_client(settings: &LlmSettings) -> Result<HiSimClient, LlmError> {
    // «Ollama 第一 → OpenAI フォールバック → 型消去 → キャッシュ» の組み立ては
    // socsim-llm の `build_live_client` に委譲する (挙動は従来の手書き実装と等価)．
    // 本ラッパは replication 固有の `LlmSettings` (cache_path) を受け取る薄い層．
    socsim_llm::build_live_client(settings.cache_path.as_deref().map(Path::new))
}

/// 任意の [`LlmClient`] (例: `mock::ScriptedClient`) をキャッシュで包んだ
/// [`HiSimClient`] を作る (主にテスト・オフラインスモーク用)．
pub fn wrap_client<C: LlmClient + 'static>(backend: C, cache: PromptCache) -> HiSimClient {
    let boxed: Box<dyn LlmClient> = Box::new(backend);
    CachingClient::new(boxed, cache)
}

/// [`LlmSettings`] から socsim-llm の [`LlmConfig`] を組み立てる．
pub fn llm_config(settings: &LlmSettings) -> LlmConfig {
    LlmConfig::deterministic()
        .with_temperature(settings.temperature)
        .with_seed(settings.seed)
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
}
