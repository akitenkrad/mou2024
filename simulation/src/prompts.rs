//! LLM プロンプト生成 (コア層 — 行動選択)．
//!
//! コアエージェント (オピニオンリーダー) の profile/memory + 当該ステップの外部
//! イベント + 隣接ノードの態度を文脈に，行動 (post/retweet/reply/like/do-nothing)
//! を選ばせるプロンプトを組む．応答は [`crate::parse::parse_action`] でパースし，
//! stance/sentiment postprocessing ([`crate::llm::classify_stance`]) で態度へ橋渡し
//! する．

/// コア層の行動選択プロンプトを組み立てる．
///
/// `profile` はコア状態の自己記述，`memory` は直近記憶 (新しい順)，`event` は
/// 当該ステップの外部イベント，`attitude` は現在の態度 $\in [-1,1]$，
/// `neighbor_attitudes` は可視隣接ノードの態度列．
pub fn action_prompt(
    profile: &str,
    memory: &[String],
    event: &str,
    attitude: f64,
    neighbor_attitudes: &[f64],
) -> String {
    let mut s = String::new();
    s.push_str(
        "You are an influential user on a social-media platform during a social movement. \
         Decide your next single action.\n\n",
    );
    s.push_str(&format!("Your profile: {profile}\n"));
    s.push_str(&format!(
        "Your current attitude toward the movement (-1=strongly against, +1=strongly in favor): {attitude:.2}\n"
    ));

    if memory.is_empty() {
        s.push_str("Your recent memory: (none)\n");
    } else {
        s.push_str("Your recent memory:\n");
        for m in memory.iter().rev().take(3) {
            s.push_str(&format!("- {m}\n"));
        }
    }

    s.push_str(&format!("\nToday's event: {event}\n"));

    if neighbor_attitudes.is_empty() {
        s.push_str("Your network is quiet right now.\n");
    } else {
        let mean: f64 = neighbor_attitudes.iter().sum::<f64>() / neighbor_attitudes.len() as f64;
        s.push_str(&format!(
            "The people you follow have an average attitude of {mean:.2} ({} of them).\n",
            neighbor_attitudes.len()
        ));
    }

    s.push_str(
        "\nThink step by step, then choose exactly one action and write a short message stating \
         your stance. Reply in this exact format:\n\
         THOUGHT: <one sentence>\n\
         ACTION: <post|retweet|reply|like|do-nothing>\n\
         MESSAGE: <one short sentence expressing your stance, or - for like/do-nothing>\n",
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_contains_event_and_attitude() {
        let p = action_prompt(
            "a journalist covering social issues",
            &["saw many posts".to_string()],
            "A celebrity shared their #MeToo story.",
            0.3,
            &[0.4, 0.5],
        );
        assert!(p.contains("#MeToo"));
        assert!(p.contains("0.30"));
        assert!(p.contains("ACTION:"));
    }

    #[test]
    fn prompt_is_deterministic() {
        let a = action_prompt("p", &[], "e", 0.0, &[0.1]);
        let b = action_prompt("p", &[], "e", 0.0, &[0.1]);
        assert_eq!(a, b);
    }
}
