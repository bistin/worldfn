//! The hardened tool loop: nested tool sets, exact tool surface, finish
//! checks, reply validation, budgets, deadlines and cancellation.
#![cfg(feature = "structured")]

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use worldfn::chat::{ChatRequest, ChatResponse, FinishReason, Message, ToolCall, ToolDefinition};
use worldfn::prelude::*;
use worldfn::{
    BoxFuture, CancelToken, LoopLimit, LoopOptions, StopReason, ToolError, ToolLoopError, ToolRun,
    Toolbox,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
struct Arg {
    n: i64,
}

#[derive(Debug, Clone, Serialize)]
struct Out {
    tool: &'static str,
    n: i64,
}

/// Nine distinct tools, as in the VM plan.
macro_rules! tools {
    ($($T:ident => $name:literal),+ $(,)?) => {$(
        struct $T;
        impl ToolSpec for $T {
            const NAME: &'static str = $name;
            const DESCRIPTION: &'static str = concat!("test tool ", $name);
            type Request = Arg;
            type Response = Out;
        }
    )+};
}

tools! {
    T1 => "t1", T2 => "t2", T3 => "t3", T4 => "t4", T5 => "t5",
    T6 => "t6", T7 => "t7", T8 => "t8", T9 => "t9",
}

type Nine = ((T1, T2, T3, T4, T5), (T6, T7, T8, T9));

fn echo<T: ToolSpec<Request = Arg, Response = Out>>() -> FakeTool<T> {
    FakeTool::new(|r: &Arg| {
        if r.n < 0 {
            Err(ToolError(format!("negative: {}", r.n)))
        } else {
            Ok(Out {
                tool: T::NAME,
                n: r.n,
            })
        }
    })
}

fn world(llm: FakeLlm) -> AgentWorld {
    let mut world = AgentWorld::new();
    world
        .provide_llm(llm)
        .unwrap()
        .provide_tool::<T1>(echo::<T1>())
        .unwrap()
        .provide_tool::<T2>(echo::<T2>())
        .unwrap()
        .provide_tool::<T3>(echo::<T3>())
        .unwrap()
        .provide_tool::<T4>(echo::<T4>())
        .unwrap()
        .provide_tool::<T5>(echo::<T5>())
        .unwrap()
        .provide_tool::<T6>(echo::<T6>())
        .unwrap()
        .provide_tool::<T7>(echo::<T7>())
        .unwrap()
        .provide_tool::<T8>(echo::<T8>())
        .unwrap()
        .provide_tool::<T9>(echo::<T9>())
        .unwrap();
    world
}

fn call(id: &str, name: &str, n: i64) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: format!(r#"{{"n":{n}}}"#),
    }
}

fn reply(text: &str, finish: FinishReason) -> ChatResponse {
    ChatResponse {
        message: Message::assistant(text),
        finish,
        usage: None,
    }
}

async fn run_nine(
    world: &AgentWorld,
    request: ChatRequest,
    options: LoopOptions,
) -> Result<ToolRun, ToolLoopError> {
    async fn agent(
        llm: Llm,
        tools: Toolbox<Nine>,
        (request, options): (Input<ChatRequest>, Input<LoopOptions>),
    ) -> Result<ToolRun, ToolLoopError> {
        tools
            .run_with(&llm, (*request).clone(), (*options).clone(), |_| {})
            .await
    }
    world
        .run_with(agent, Scope::of(request).with(options))
        .await
        .expect("bindings are complete")
}

#[tokio::test]
async fn nine_tools_in_nested_sets_dispatch_and_show_in_the_signature() -> TestResult {
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([
            call("a", "t1", 1),
            call("b", "t5", 5),
            call("c", "t6", 6),
            call("d", "t9", 9),
        ]))
        .then_answer("done");
    let world = world(llm.clone());
    let run = run_nine(&world, ChatRequest::prompt("go"), LoopOptions::new(3)).await?;
    let contents: Vec<_> = run.calls.iter().map(|(_, r)| r.content.as_str()).collect();
    assert_eq!(
        contents,
        [
            r#"{"tool":"t1","n":1}"#,
            r#"{"tool":"t5","n":5}"#,
            r#"{"tool":"t6","n":6}"#,
            r#"{"tool":"t9","n":9}"#
        ]
    );
    let advertised: Vec<_> = llm.requests()[0]
        .tools
        .iter()
        .map(|t| t.name.clone())
        .collect();
    assert_eq!(
        advertised,
        ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
    );

    async fn sig(_tools: Toolbox<Nine>) {}
    let meta = sig.into_agent().meta().to_string();
    assert_eq!(meta.matches("can call Tool<").count(), 9, "{meta}");

    // A missing handler deep in the nesting is reported before the body runs.
    let mut partial = AgentWorld::new();
    partial.provide_tool::<T1>(echo::<T1>())?;
    let err = partial.prepare(sig).err().unwrap().to_string();
    assert!(err.contains("✓ Tool<t1>"), "{err}");
    assert!(err.contains("✗ Tool<t9>: no provider registered"), "{err}");
    Ok(())
}

#[tokio::test]
async fn the_model_sees_exactly_the_toolbox() -> TestResult {
    let llm = FakeLlm::new().then_answer("ok");
    let world = world(llm.clone());
    let request = ChatRequest::prompt("go").tool(ToolDefinition {
        name: "place_order".into(),
        description: "not in the signature".into(),
        parameters: "{}".into(),
    });
    run_nine(&world, request, LoopOptions::new(1)).await?;
    let tools = &llm.requests()[0].tools;
    assert_eq!(tools.len(), 9);
    assert!(tools.iter().all(|t| t.name != "place_order"));
    Ok(())
}

#[tokio::test]
async fn duplicate_tool_names_are_rejected() {
    struct Again;
    impl ToolSpec for Again {
        const NAME: &'static str = "t1";
        type Request = Arg;
        type Response = Out;
    }
    async fn agent(llm: Llm, tools: Toolbox<(T1, Again)>) -> Result<ToolRun, ToolLoopError> {
        tools.run(&llm, ChatRequest::prompt("go"), 1, |_| {}).await
    }
    let mut world = world(FakeLlm::new().then_answer("never sent"));
    world
        .provide_tool::<Again>(FakeTool::new(|_: &Arg| Err(ToolError("x".into()))))
        .unwrap();
    let err = world.run(agent).await.unwrap().unwrap_err();
    assert!(
        matches!(&err, ToolLoopError::InvalidToolSurface(m) if m.contains("`t1`")),
        "{err}"
    );
}

#[tokio::test]
async fn only_a_complete_answer_ends_the_run() -> TestResult {
    let cases = [
        (reply("cut off mid-sent", FinishReason::Length), false),
        (reply("", FinishReason::ContentFilter), false),
        (reply("  ", FinishReason::Stop), false),
        // Claims tool calls but has none.
        (reply("calling…", FinishReason::ToolCalls), false),
        (reply("fine", FinishReason::Other), true),
        (reply("fine", FinishReason::Stop), true),
    ];
    for (response, ok) in cases {
        let finish = response.finish;
        let world = world(FakeLlm::new().then_response(response));
        let result = run_nine(&world, ChatRequest::prompt("go"), LoopOptions::new(2)).await;
        match result {
            Ok(_) => assert!(ok, "{finish:?} accepted"),
            Err(ToolLoopError::Unfinished { response, .. }) => {
                assert!(!ok, "{finish:?} rejected");
                assert_eq!(response.finish, finish);
            }
            Err(other) => panic!("{finish:?}: {other}"),
        }
    }
    Ok(())
}

#[tokio::test]
async fn the_transcript_ends_with_the_answer() -> TestResult {
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([call("a", "t2", 2)]))
        .then_answer("two");
    let world = world(llm);
    let run = run_nine(&world, ChatRequest::prompt("go"), LoopOptions::new(3)).await?;
    let roles: Vec<_> = run.request.messages.iter().map(|m| m.role).collect();
    use worldfn::chat::MessageRole::*;
    assert_eq!(roles, [User, Assistant, Tool, Assistant]);
    assert_eq!(run.request.messages.last().unwrap().text(), "two");
    Ok(())
}

#[tokio::test]
async fn unsafe_replies_run_nothing() -> TestResult {
    for (calls, expected) in [
        (vec![call("x", "t1", 1), call("x", "t2", 2)], "used twice"),
        (vec![call("", "t1", 1)], "no id"),
    ] {
        let world = world(FakeLlm::new().then_response(ChatResponse::tool_calls(calls)));
        let err = run_nine(&world, ChatRequest::prompt("go"), LoopOptions::new(2))
            .await
            .unwrap_err();
        match err {
            ToolLoopError::InvalidTurn { reason, calls } => {
                assert!(reason.contains(expected), "{reason}");
                assert!(calls.is_empty());
            }
            other => panic!("{other}"),
        }
    }
    Ok(())
}

#[tokio::test]
async fn budgets_stop_the_run_with_its_calls() -> TestResult {
    // Calls in one reply.
    let w = world(FakeLlm::new().then_response(ChatResponse::tool_calls([
        call("a", "t1", 1),
        call("b", "t2", 2),
        call("c", "t3", 3),
    ])));
    let err = run_nine(
        &w,
        ChatRequest::prompt("go"),
        LoopOptions::new(5).max_calls_per_turn(2),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ToolLoopError::LimitExceeded {
            limit: LoopLimit::CallsPerTurn,
            ..
        }
    ));

    // Calls over the run: the third is not executed.
    let llm = FakeLlm::responding_to(|request| {
        let n = request.messages.len() as i64;
        ChatResponse::tool_calls([call(&format!("c{n}"), "t1", n)])
    });
    let w = world(llm);
    let err = run_nine(
        &w,
        ChatRequest::prompt("go"),
        LoopOptions::new(10).max_tool_calls(2),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ToolLoopError::LimitExceeded {
            limit: LoopLimit::ToolCalls,
            ..
        }
    ));
    assert_eq!(err.calls().len(), 2);

    // The same failing call, again and again.
    let llm = FakeLlm::responding_to(|request| {
        let n = request.messages.len();
        ChatResponse::tool_calls([call(&format!("c{n}"), "t1", -1)])
    });
    let err = run_nine(
        &world_with(llm),
        ChatRequest::prompt("go"),
        LoopOptions::new(10).max_repeated_failures(3),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ToolLoopError::LimitExceeded {
            limit: LoopLimit::RepeatedFailures,
            ..
        }
    ));
    assert_eq!(err.calls().len(), 3);
    assert!(err.calls().iter().all(|(_, r)| r.is_error));

    // Different failures add up.
    let llm = FakeLlm::responding_to(|request| {
        let n = request.messages.len() as i64;
        ChatResponse::tool_calls([call(&format!("c{n}"), "t1", -n)])
    });
    let err = run_nine(
        &world_with(llm),
        ChatRequest::prompt("go"),
        LoopOptions::new(10)
            .max_failed_calls(2)
            .max_repeated_failures(3),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        err,
        ToolLoopError::LimitExceeded {
            limit: LoopLimit::FailedCalls,
            ..
        }
    ));
    assert_eq!(err.calls().len(), 3);
    Ok(())
}

fn world_with(llm: FakeLlm) -> AgentWorld {
    world(llm)
}

#[tokio::test]
async fn long_results_are_cut_with_a_note() -> TestResult {
    struct Big;
    impl ToolSpec for Big {
        const NAME: &'static str = "big";
        type Request = Arg;
        type Response = String;
    }
    async fn agent(llm: Llm, tools: Toolbox<Big>) -> Result<ToolRun, ToolLoopError> {
        tools
            .run_with(
                &llm,
                ChatRequest::prompt("go"),
                LoopOptions::new(2).max_result_bytes(40),
                |_| {},
            )
            .await
    }
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([call("a", "big", 1)]))
        .then_answer("ok");
    let mut world = AgentWorld::new();
    world
        .provide_llm(llm.clone())?
        .provide_tool::<Big>(FakeTool::new(|_: &Arg| Ok("é".repeat(100))))?;
    let run = world.run(agent).await??;
    let content = &run.calls[0].1.content;
    assert!(content.starts_with("\"éé"), "{content}");
    assert!(
        content.ends_with("[truncated: showing 39 of 202 bytes]"),
        "{content}"
    );
    Ok(())
}

/// A tool that never finishes, and optionally cancels a token first.
struct Hang;
impl ToolSpec for Hang {
    const NAME: &'static str = "hang";
    type Request = Arg;
    type Response = Out;
}

struct HangHandler(Option<CancelToken>);
impl worldfn::ToolHandler<Hang> for HangHandler {
    fn call(&self, _: Arg) -> BoxFuture<'_, Result<Out, ToolError>> {
        if let Some(token) = &self.0 {
            token.cancel();
        }
        Box::pin(std::future::pending())
    }
}

async fn hang_run(handler: HangHandler, options: LoopOptions) -> ToolLoopError {
    async fn agent(
        llm: Llm,
        tools: Toolbox<(T1, Hang)>,
        options: Input<LoopOptions>,
    ) -> Result<ToolRun, ToolLoopError> {
        tools
            .run_with(&llm, ChatRequest::prompt("go"), (*options).clone(), |_| {})
            .await
    }
    let llm = FakeLlm::new().then_response(ChatResponse::tool_calls([
        call("a", "t1", 1),
        call("b", "hang", 2),
        call("c", "t1", 3),
    ]));
    let mut world = world(llm);
    world.provide_tool::<Hang>(handler).unwrap();
    world
        .run_with(agent, Scope::of(options))
        .await
        .unwrap()
        .unwrap_err()
}

#[tokio::test]
async fn a_deadline_abandons_a_stuck_tool() {
    let started = Instant::now();
    let err = hang_run(
        HangHandler(None),
        LoopOptions::new(3).timeout(Duration::from_millis(100)),
    )
    .await;
    assert!(started.elapsed() < Duration::from_secs(5));
    match err {
        ToolLoopError::Stopped { reason, calls } => {
            assert_eq!(reason, StopReason::DeadlineExceeded);
            // The call before the stuck one completed; nothing after it ran.
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].0.id, "a");
        }
        other => panic!("{other}"),
    }
}

#[tokio::test]
async fn cancelling_stops_the_run() {
    let token = CancelToken::new();
    let err = hang_run(
        HangHandler(Some(token.clone())),
        LoopOptions::new(3).cancel(token),
    )
    .await;
    assert!(
        matches!(
            err,
            ToolLoopError::Stopped {
                reason: StopReason::Cancelled,
                ..
            }
        ),
        "{err}"
    );

    // Already cancelled: the model is never called.
    let token = CancelToken::new();
    token.cancel();
    let llm = FakeLlm::new().then_answer("unused");
    let world = world(llm.clone());
    let err = run_nine(
        &world,
        ChatRequest::prompt("go"),
        LoopOptions::new(2).cancel(token),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, ToolLoopError::Stopped { .. }));
    assert_eq!(llm.calls(), 0);
}

#[tokio::test]
async fn a_failed_model_call_keeps_the_calls_that_already_ran() {
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([
            call("a", "t1", 1),
            call("b", "t2", 2),
        ]))
        .then_error("upstream 503");
    let w = world(llm);
    let err = run_nine(&w, ChatRequest::prompt("go"), LoopOptions::new(3))
        .await
        .unwrap_err();
    match &err {
        ToolLoopError::Llm { error, calls } => {
            assert!(error.0.contains("503"));
            let ids: Vec<_> = calls.iter().map(|(c, _)| c.id.as_str()).collect();
            assert_eq!(ids, ["a", "b"]);
        }
        other => panic!("{other}"),
    }
    assert!(err.to_string().contains("after 2 tool call(s)"), "{err}");
}

#[tokio::test]
async fn calls_in_a_cut_off_or_filtered_reply_never_run() {
    for finish in [FinishReason::Length, FinishReason::ContentFilter] {
        let llm = FakeLlm::new().then_response(ChatResponse {
            finish,
            ..ChatResponse::tool_calls([call("a", "t1", 1)])
        });
        let w = world(llm);
        let err = run_nine(&w, ChatRequest::prompt("go"), LoopOptions::new(3))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, ToolLoopError::Unfinished { calls, .. } if calls.is_empty()),
            "{finish:?}: {err}"
        );
    }
}

#[tokio::test]
async fn the_observer_sees_the_full_result_even_when_the_model_sees_a_cut() -> TestResult {
    use std::sync::{Arc, Mutex};
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let observer = move |_: &ToolCall, result: &worldfn::chat::ToolResult| {
        sink.lock().unwrap().push(result.content.clone());
        Vec::new()
    };
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([call("a", "t1", 123_456_789)]))
        .then_answer("ok");
    let w = world(llm);
    let run = run_nine(
        &w,
        ChatRequest::prompt("go"),
        LoopOptions::new(3).max_result_bytes(10).observer(observer),
    )
    .await?;
    assert!(run.calls[0].1.content.contains("[truncated"));
    assert_eq!(*seen.lock().unwrap(), [r#"{"tool":"t1","n":123456789}"#]);
    Ok(())
}
