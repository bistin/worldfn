//! Model-driven tool calling: `Toolbox<(A, B)>` and its bounded loop.
#![cfg(feature = "structured")]

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use worldfn::chat::{ChatRequest, ChatResponse, MessageRole, Part, ToolCall, Usage};
use worldfn::prelude::*;
use worldfn::{LoopEvent, ToolLoopError, Toolbox};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Deterministic calculation: the model must not do this arithmetic itself.
struct GrowthRate;

#[derive(Debug, Clone, PartialEq, Deserialize, schemars::JsonSchema)]
struct GrowthInput {
    /// This period's value.
    current: f64,
    /// The comparison period's value.
    previous: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct GrowthOutput {
    rate_pct: f64,
}

impl ToolSpec for GrowthRate {
    const NAME: &'static str = "growth_rate";
    const DESCRIPTION: &'static str = "Percentage growth from previous to current.";
    type Request = GrowthInput;
    type Response = GrowthOutput;
}

struct Quote;

#[derive(Debug, Clone, PartialEq, Deserialize, schemars::JsonSchema)]
struct QuoteInput {
    /// MARKET:SYMBOL, e.g. TWSE:2330.
    instrument: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct QuoteOutput {
    price: f64,
    as_of: String,
    source: String,
}

impl ToolSpec for Quote {
    const NAME: &'static str = "quote";
    const DESCRIPTION: &'static str = "Delayed quote with its source and time.";
    type Request = QuoteInput;
    type Response = QuoteOutput;
}

fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments.into(),
    }
}

struct Fixture {
    world: AgentWorld,
    growth: FakeTool<GrowthRate>,
    quote: FakeTool<Quote>,
}

fn fixture(llm: FakeLlm) -> Fixture {
    let growth = FakeTool::<GrowthRate>::new(|r| {
        if r.previous == 0.0 {
            return Err(worldfn::ToolError("previous value is zero".into()));
        }
        Ok(GrowthOutput {
            rate_pct: (r.current - r.previous) / r.previous * 100.0,
        })
    });
    let quote = FakeTool::<Quote>::new(|r| {
        Ok(QuoteOutput {
            price: 612.0,
            as_of: "2026-09-30T13:30:00+08:00".into(),
            source: format!("fixture:{}", r.instrument),
        })
    });
    let mut world = AgentWorld::new();
    world
        .provide_llm(llm)
        .unwrap()
        .provide_tool::<GrowthRate>(growth.clone())
        .unwrap()
        .provide_tool::<Quote>(quote.clone())
        .unwrap();
    Fixture {
        world,
        growth,
        quote,
    }
}

/// The agent: the loop is an explicit, bounded call in its body.
async fn analyst(
    task: Input<Task>,
    llm: Llm,
    tools: Toolbox<(GrowthRate, Quote)>,
) -> Result<worldfn::ToolRun, ToolLoopError> {
    tools
        .run(
            &llm,
            ChatRequest::new()
                .system("Use tools for every number.")
                .user(task.0.clone()),
            4,
            |_| {},
        )
        .await
}

#[tokio::test]
async fn model_calls_tools_and_numbers_come_from_code() -> TestResult {
    let llm = FakeLlm::new()
        .then_response(ChatResponse {
            usage: Some(Usage {
                input_tokens: 100,
                output_tokens: 10,
            }),
            ..ChatResponse::tool_calls([
                call("c1", "growth_rate", r#"{"current":120,"previous":100}"#),
                call("c2", "quote", r#"{"instrument":"TWSE:2330"}"#),
            ])
        })
        .then_response(ChatResponse {
            usage: Some(Usage {
                input_tokens: 150,
                output_tokens: 20,
            }),
            ..ChatResponse::text("Revenue grew 20%; the delayed quote is 612.")
        });
    let f = fixture(llm.clone());

    let run = f
        .world
        .run_with(analyst, Scope::of(Task::new("How did revenue grow?")))
        .await??;

    assert_eq!(
        run.response.message.text(),
        "Revenue grew 20%; the delayed quote is 612."
    );
    assert_eq!(run.steps, 2);
    assert_eq!(
        run.usage,
        Usage {
            input_tokens: 250,
            output_tokens: 30
        }
    );
    assert_eq!(run.calls.len(), 2);
    assert_eq!(run.calls[0].1.content, r#"{"rate_pct":20.0}"#);
    assert!(!run.calls[0].1.is_error);
    assert_eq!((f.growth.calls(), f.quote.calls()), (1, 1));

    // First request: tool definitions with schemas generated from the types.
    let requests = llm.requests();
    let tools = &requests[0].tools;
    assert_eq!(
        tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        ["growth_rate", "quote"]
    );
    let schema: serde_json::Value = serde_json::from_str(&tools[0].parameters)?;
    assert_eq!(schema["properties"]["current"]["type"], "number");
    assert_eq!(
        tools[1].description,
        "Delayed quote with its source and time."
    );

    // Second request: the model's calls and their results were sent back.
    let second = &requests[1];
    assert_eq!(second.messages.len(), 4);
    assert_eq!(second.messages[1].role, MessageRole::Assistant);
    assert_eq!(second.messages[1].tool_calls().count(), 2);
    assert!(matches!(&second.messages[2].parts[0], Part::ToolResult(r) if r.call_id == "c1"));
    assert!(matches!(&second.messages[3].parts[0], Part::ToolResult(r) if r.call_id == "c2"));
    Ok(())
}

#[tokio::test]
async fn tools_outside_the_signature_are_refused() -> TestResult {
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([call(
            "c1",
            "place_order",
            r#"{"instrument":"TWSE:2330","side":"buy"}"#,
        )]))
        .then_answer("I can't place orders; here is what to check instead.");
    let f = fixture(llm);

    let run = f
        .world
        .run_with(analyst, Scope::of(Task::new("Buy 2330 for me")))
        .await??;
    let (_, result) = &run.calls[0];
    assert!(result.is_error);
    assert_eq!(
        result.content,
        "unknown tool `place_order`; available: growth_rate, quote"
    );
    assert_eq!((f.growth.calls(), f.quote.calls()), (0, 0));
    Ok(())
}

#[tokio::test]
async fn bad_arguments_and_tool_errors_go_back_to_the_model() -> TestResult {
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([
            call("c1", "growth_rate", r#"{"current":"lots","previous":1}"#),
            call("c2", "growth_rate", "not json"),
            call("c3", "growth_rate", r#"{"current":5,"previous":0}"#),
        ]))
        .then_answer("I could not compute the growth rate.");
    let f = fixture(llm);

    let run = f
        .world
        .run_with(analyst, Scope::of(Task::new("growth?")))
        .await??;
    let results: Vec<_> = run.calls.iter().map(|(_, r)| r).collect();
    assert!(results.iter().all(|r| r.is_error));
    assert!(
        results[0].content.starts_with("invalid arguments:"),
        "{}",
        results[0].content
    );
    assert!(
        results[1].content.starts_with("invalid arguments:"),
        "{}",
        results[1].content
    );
    assert_eq!(results[2].content, "previous value is zero");
    // Invalid arguments never reach the handler; the zero case did.
    assert_eq!(f.growth.calls(), 1);
    Ok(())
}

#[tokio::test]
async fn the_loop_is_bounded() {
    let llm = FakeLlm::responding_to(|_| {
        ChatResponse::tool_calls([call("c", "quote", r#"{"instrument":"TWSE:2330"}"#)])
    });
    let f = fixture(llm);
    let err = f
        .world
        .run_with(analyst, Scope::of(Task::new("loop forever")))
        .await
        .unwrap()
        .unwrap_err();
    let ToolLoopError::MaxSteps { max_steps, calls } = err else {
        panic!("{err:?}")
    };
    assert_eq!((max_steps, calls.len()), (4, 4));
    assert_eq!(f.quote.calls(), 4);
}

#[tokio::test]
async fn non_object_requests_are_wrapped_for_the_model() -> TestResult {
    async fn searcher(llm: Llm, tools: Toolbox<(WebSearch,)>) -> Result<usize, ToolLoopError> {
        let run = tools
            .run(&llm, ChatRequest::prompt("search"), 2, |_| {})
            .await?;
        Ok(run.calls.len())
    }
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([call(
            "c1",
            "web_search",
            r#"{"input":"bevy system params"}"#,
        )]))
        .then_answer("done");
    let web = FakeTool::<WebSearch>::with_response(vec![]);
    let mut world = AgentWorld::new();
    world
        .provide_llm(llm.clone())?
        .provide_tool::<WebSearch>(web.clone())?;

    assert_eq!(world.run(searcher).await??, 1);
    assert_eq!(web.requests(), ["bevy system params"]);
    let schema: serde_json::Value = serde_json::from_str(&llm.requests()[0].tools[0].parameters)?;
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["input"]["type"], "string");
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
enum Progress {
    Calling(String),
    Done(String, bool),
}

#[tokio::test]
async fn loop_events_can_be_streamed_through_emit() -> TestResult {
    async fn streamed(
        llm: Llm,
        tools: Toolbox<(GrowthRate, Quote)>,
        events: Emit<Progress>,
    ) -> Result<String, ToolLoopError> {
        let run = tools
            .run(
                &llm,
                ChatRequest::prompt("growth?"),
                3,
                |event| match event {
                    LoopEvent::ToolCall(call) => {
                        events.send(Progress::Calling(call.name.clone()));
                    }
                    LoopEvent::ToolResult(call, result) => {
                        events.send(Progress::Done(call.name.clone(), result.is_error));
                    }
                },
            )
            .await?;
        Ok(run.response.message.text())
    }
    let llm = FakeLlm::new()
        .then_response(ChatResponse::tool_calls([call(
            "c1",
            "growth_rate",
            r#"{"current":2,"previous":1}"#,
        )]))
        .then_answer("100%");
    let f = fixture(llm);
    let (emitter, mut rx) = worldfn::emit::channel::<Progress>();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let answer = f
        .world
        .run_with(streamed, Scope::new().with(emitter))
        .await??;
    while let Some(event) = rx.recv().await {
        seen.lock().unwrap().push(event);
    }
    assert_eq!(answer, "100%");
    assert_eq!(
        *seen.lock().unwrap(),
        [
            Progress::Calling("growth_rate".into()),
            Progress::Done("growth_rate".into(), false)
        ]
    );
    Ok(())
}

#[test]
fn signature_lists_the_tool_surface_and_missing_handlers() {
    assert_eq!(
        analyst.into_agent().meta().to_string(),
        "analyst\n\
         ├── reads Input<Task>\n\
         ├── requires Llm\n\
         ├── can call Tool<growth_rate>\n\
         └── can call Tool<quote>"
    );
    let mut world = AgentWorld::new();
    world
        .provide_llm(FakeLlm::new())
        .unwrap()
        .provide_tool::<Quote>(FakeTool::new(|_| Err(worldfn::ToolError("unused".into()))))
        .unwrap();
    let err = world.prepare(analyst).err().unwrap();
    assert!(
        err.to_string()
            .contains("✗ Tool<growth_rate>: no provider registered"),
        "{err}"
    );
    assert!(err.to_string().contains("✓ Tool<quote>"), "{err}");
}
