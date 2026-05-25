//! LLM 応答パース (コア層の行動決定)．
//!
//! [`crate::prompts::action_prompt`] の応答テキストから `ACTION` / `MESSAGE` を
//! 抽出する．読めない量は安全側 (`ActionKind::DoNothing` / 空メッセージ) に倒す．

/// コアエージェントが選んだ行動の種別．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    /// 新規投稿 (自分の態度を発信)．
    Post,
    /// リツイート (call-to-action を増幅)．
    Retweet,
    /// 返信 (態度を発信)．
    Reply,
    /// いいね (弱い同調シグナル)．
    Like,
    /// 何もしない．
    DoNothing,
}

impl ActionKind {
    /// 文字列トークンから行動種別をパースする (未知語は `DoNothing`)．
    pub fn parse(s: &str) -> ActionKind {
        match s.trim().to_ascii_lowercase().as_str() {
            "post" | "tweet" => ActionKind::Post,
            "retweet" | "repost" | "share" => ActionKind::Retweet,
            "reply" | "comment" => ActionKind::Reply,
            "like" | "upvote" => ActionKind::Like,
            _ => ActionKind::DoNothing,
        }
    }

    /// 短いラベル (出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            ActionKind::Post => "post",
            ActionKind::Retweet => "retweet",
            ActionKind::Reply => "reply",
            ActionKind::Like => "like",
            ActionKind::DoNothing => "do-nothing",
        }
    }

    /// この行動がメッセージ (態度シグナル) を発信するか．
    ///
    /// post/retweet/reply はネットワークへ態度を伝える (周辺層への影響源)．
    /// like/do-nothing は発信しない (周辺層への影響無し)．
    pub fn broadcasts(&self) -> bool {
        matches!(
            self,
            ActionKind::Post | ActionKind::Retweet | ActionKind::Reply
        )
    }
}

/// パース済みの行動決定．
#[derive(Debug, Clone)]
pub struct ActionDecision {
    /// 行動種別．
    pub kind: ActionKind,
    /// 発信メッセージ本文 (post/retweet/reply 用; stance 分類の入力)．
    pub message: Option<String>,
}

/// `key:` で始まる行の値を取り出す (先頭一致，大文字小文字無視)．
fn field<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let klower = key.to_ascii_lowercase();
    for line in text.lines() {
        let line = line.trim();
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix(&klower) {
            let offset = line.len() - rest.len();
            return Some(line[offset..].trim_start_matches(':').trim());
        }
    }
    None
}

/// 応答テキストから行動決定をパースする．
pub fn parse_action(text: &str) -> ActionDecision {
    let kind = field(text, "ACTION")
        .map(ActionKind::parse)
        .unwrap_or(ActionKind::DoNothing);

    let message = field(text, "MESSAGE").and_then(|s| {
        let s = s.trim();
        if s.is_empty() || s == "-" {
            None
        } else {
            Some(s.to_string())
        }
    });

    ActionDecision { kind, message }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_post_with_message() {
        let t = "THOUGHT: I will speak.\nACTION: post\nMESSAGE: I support this movement.";
        let d = parse_action(t);
        assert_eq!(d.kind, ActionKind::Post);
        assert_eq!(d.message.as_deref(), Some("I support this movement."));
        assert!(d.kind.broadcasts());
    }

    #[test]
    fn parses_like_no_message() {
        let d = parse_action("ACTION: like\nMESSAGE: -");
        assert_eq!(d.kind, ActionKind::Like);
        assert_eq!(d.message, None);
        assert!(!d.kind.broadcasts());
    }

    #[test]
    fn unknown_action_is_do_nothing() {
        let d = parse_action("ACTION: ponder");
        assert_eq!(d.kind, ActionKind::DoNothing);
    }

    #[test]
    fn case_insensitive() {
        let d = parse_action("action: retweet\nmessage: amplify the call");
        assert_eq!(d.kind, ActionKind::Retweet);
        assert_eq!(d.message.as_deref(), Some("amplify the call"));
    }
}
