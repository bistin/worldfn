//! Message-level LLM API and typed (structured) replies, against `FakeLlm`.

use worldfn::LlmError;
use worldfn::chat::{
    ChatRequest, ChatResponse, FinishReason, Message, MessageRole, OutputFormat, Part, ToolCall,
    ToolResult,
};
use worldfn::prelude::*;

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[tokio::test]
async fn chat_sends_the_whole_request_and_returns_the_whole_reply() -> TestResult {
    let fake = FakeLlm::new().then_response(ChatResponse::tool_calls([ToolCall {
        id: "call_1".into(),
        name: "quote".into(),
        arguments: r#"{"instrument":"TWSE:2330"}"#.into(),
    }]));
    let llm = Llm::new(fake.clone());

    let request = ChatRequest::new()
        .system("You are a mentor.")
        .user("What did I say about 2330?")
        .assistant("You said you'd wait for earnings.")
        .user("And the price now?");
    let reply = llm.chat(request.clone()).await?;

    assert_eq!(reply.finish, FinishReason::ToolCalls);
    let calls: Vec<_> = reply.message.tool_calls().collect();
    assert_eq!(calls[0].name, "quote");
    assert_eq!(fake.requests(), [request]);
    assert_eq!(fake.prompts(), ["And the price now?"]);
    Ok(())
}

#[tokio::test]
async fn complete_is_a_single_user_message() -> TestResult {
    let fake = FakeLlm::echo();
    let llm = Llm::new(fake.clone());
    assert_eq!(llm.complete("hi").await?, "hi");
    assert_eq!(fake.requests(), [ChatRequest::prompt("hi")]);
    Ok(())
}

#[test]
fn tool_results_are_messages_too() {
    let message = Message::tool_result(ToolResult {
        call_id: "call_1".into(),
        content: "612.0".into(),
        is_error: false,
    });
    assert_eq!(message.role, MessageRole::Tool);
    assert!(matches!(message.parts[0], Part::ToolResult(_)));
    assert_eq!(message.text(), "");
}

// ---- Structured output -----------------------------------------------------

#[cfg(feature = "structured")]
mod structured {
    use super::*;
    use worldfn::StructuredError;

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    #[serde(rename_all = "snake_case")]
    enum Verdict {
        FollowedStrategy,
        Deviated,
    }

    #[derive(Debug, PartialEq, serde::Deserialize, schemars::JsonSchema)]
    struct ReviewNote {
        verdict: Verdict,
        missing_evidence: Vec<String>,
    }

    #[tokio::test]
    async fn complete_as_sends_a_schema_and_parses_the_reply() -> TestResult {
        let fake = FakeLlm::with_answer(
            "```json\n{\"verdict\":\"deviated\",\"missing_evidence\":[\"Q3 margin\"]}\n```",
        );
        let note: ReviewNote = Llm::new(fake.clone())
            .complete_as(ChatRequest::prompt("Review my trade."))
            .await?;
        assert_eq!(
            note,
            ReviewNote {
                verdict: Verdict::Deviated,
                missing_evidence: vec!["Q3 margin".into()],
            }
        );

        let OutputFormat::Json { name, schema } = &fake.requests()[0].output else {
            panic!("expected a JSON output format");
        };
        assert_eq!(name, "ReviewNote");
        let schema: serde_json::Value = serde_json::from_str(schema)?;
        assert!(schema.pointer("/properties/verdict").is_some(), "{schema}");
        assert!(
            schema.pointer("/properties/missing_evidence").is_some(),
            "{schema}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn invalid_output_is_corrected_once_by_default() -> TestResult {
        let fake = FakeLlm::new()
            .then_answer("It looks like you deviated.")
            .then_answer(r#"{"verdict":"followed_strategy","missing_evidence":[]}"#);
        let note: ReviewNote = Llm::new(fake.clone())
            .complete_as(ChatRequest::prompt("Review my trade."))
            .await?;
        assert_eq!(note.verdict, Verdict::FollowedStrategy);

        // The retry shows the model its reply and why it was rejected.
        let retry = &fake.requests()[1];
        assert_eq!(retry.messages.len(), 3);
        assert_eq!(retry.messages[1].text(), "It looks like you deviated.");
        assert!(
            retry.messages[2]
                .text()
                .starts_with("That reply was not valid:"),
            "{:?}",
            retry.messages[2]
        );
        Ok(())
    }

    #[tokio::test]
    async fn gives_up_with_the_last_reply_after_retries() {
        let fake = FakeLlm::scripted(["nope", r#"{"verdict":"maybe","missing_evidence":[]}"#]);
        let err = Llm::new(fake)
            .complete_as::<ReviewNote>(ChatRequest::prompt("x"))
            .await
            .unwrap_err();
        let StructuredError::Invalid {
            reason,
            raw,
            attempts,
        } = err
        else {
            panic!("{err:?}")
        };
        assert_eq!(attempts, 2);
        assert_eq!(raw, r#"{"verdict":"maybe","missing_evidence":[]}"#);
        assert!(reason.contains("unknown variant `maybe`"), "{reason}");
    }

    #[tokio::test]
    async fn truncated_replies_are_not_parsed_and_llm_errors_pass_through() {
        let cut_off = ChatResponse {
            finish: FinishReason::Length,
            ..ChatResponse::text(r#"{"verdict":"deviated","missing_evidence":["a"]}"#)
        };
        let err = Llm::new(FakeLlm::new().then_response(cut_off))
            .complete_as_retrying::<ReviewNote>(ChatRequest::prompt("x"), 0)
            .await
            .unwrap_err();
        assert!(
            matches!(&err, StructuredError::Invalid { reason, .. } if reason.contains("cut off"))
        );

        let err = Llm::new(FakeLlm::new().then_error("rate limited"))
            .complete_as::<ReviewNote>(ChatRequest::prompt("x"))
            .await
            .unwrap_err();
        assert_eq!(err, StructuredError::Llm(LlmError("rate limited".into())));
    }

    #[tokio::test]
    async fn typed_replies_inside_an_agent() -> TestResult {
        async fn reviewer(task: Input<Task>, llm: Llm) -> Result<ReviewNote, StructuredError> {
            llm.complete_as(
                ChatRequest::new()
                    .system("You review decisions; never give buy or sell advice.")
                    .user(task.0.clone()),
            )
            .await
        }
        let mut world = AgentWorld::new();
        world.provide_llm(FakeLlm::with_answer(
            r#"{"verdict":"followed_strategy","missing_evidence":[]}"#,
        ))?;
        let note = world
            .run_with(reviewer, Scope::of(Task::new("review dec_1")))
            .await??;
        assert_eq!(note.verdict, Verdict::FollowedStrategy);
        Ok(())
    }
}
