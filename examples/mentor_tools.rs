//! Tool calling in a mentor-style agent: numbers come from code, the model
//! only explains them, and it can only call the tools in its signature.
//!
//! ```sh
//! cargo run --example mentor_tools                      # scripted fake model
//! WORLDFN_PROVIDER=deepseek WORLDFN_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=sk-... \
//!   cargo run --example mentor_tools --features openai-compat -- "月營收從 2.8 億成長到 3.2 億,這算強嗎?"
//! ```

use serde::{Deserialize, Serialize};
use worldfn::chat::{ChatRequest, ChatResponse, MessageRole, ToolCall};
use worldfn::prelude::*;
use worldfn::{LoopEvent, ToolLoopError, Toolbox};

mod common;

/// Growth between two periods, computed deterministically.
struct GrowthRate;

#[derive(Deserialize, schemars::JsonSchema)]
struct GrowthInput {
    /// Value in the current period.
    current: f64,
    /// Value in the comparison period.
    previous: f64,
}

#[derive(Serialize)]
struct GrowthOutput {
    rate_pct: f64,
}

impl ToolSpec for GrowthRate {
    const NAME: &'static str = "growth_rate";
    const DESCRIPTION: &'static str =
        "Percentage change from `previous` to `current`. Use it for every growth figure.";
    type Request = GrowthInput;
    type Response = GrowthOutput;
}

const MENTOR: &str = "You are an investment mentor. Never recommend buying or selling \
and never predict prices. Use the growth_rate tool for every growth figure instead of \
computing it yourself. Then ask the user what evidence would prove their thesis wrong.";

async fn mentor(
    question: Input<Task>,
    llm: Llm,
    tools: Toolbox<(GrowthRate,)>,
) -> Result<String, ToolLoopError> {
    let run = tools
        .run(
            &llm,
            ChatRequest::new().system(MENTOR).user(question.0.clone()),
            4,
            |event| match event {
                LoopEvent::ToolCall(call) => println!("  → {}({})", call.name, call.arguments),
                LoopEvent::ToolResult(_, result) => println!("  ← {}", result.content),
                LoopEvent::ModelResponded {
                    step,
                    usage: Some(usage),
                } => println!("\n  [model call {step}] {usage}"),
                // The answer, printed as it is generated.
                LoopEvent::Text(text) => {
                    print!("{text}");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
                _ => {}
            },
        )
        .await?;
    println!("\n  ({} model call(s))", run.steps);
    if run.usage != Default::default() {
        println!("  [total] {}", run.usage);
    }
    Ok(run.response.message.text())
}

/// Without a provider: call the tool once, then answer from its result.
fn scripted_model() -> FakeLlm {
    FakeLlm::responding_to(|request| {
        let tool_output = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::Tool)
            .map(|m| match &m.parts[0] {
                worldfn::chat::Part::ToolResult(r) => r.content.clone(),
                _ => String::new(),
            });
        match tool_output {
            None => ChatResponse::tool_calls([ToolCall {
                id: "call_1".into(),
                name: "growth_rate".into(),
                arguments: r#"{"current":3.2,"previous":2.8}"#.into(),
            }]),
            Some(output) => ChatResponse::text(format!(
                "(fake) The tool computed {output}. That number alone doesn't say whether \
                 your thesis holds: which result next quarter would show you were wrong?"
            )),
        }
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let question = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let question = if question.is_empty() {
        "Monthly revenue went from 2.8 to 3.2 (hundred million). Is that strong?".to_owned()
    } else {
        question
    };

    let llm = common::llm_from_env(MENTOR)?.unwrap_or_else(|| {
        println!("(WORLDFN_PROVIDER unset: using a scripted fake model)\n");
        Llm::new(scripted_model())
    });
    let mut world = AgentWorld::new();
    world
        .provide(llm)?
        .provide_tool::<GrowthRate>(FakeTool::new(|r: &GrowthInput| {
            Ok(GrowthOutput {
                rate_pct: ((r.current - r.previous) / r.previous * 10_000.0).round() / 100.0,
            })
        }))?;

    println!("{}\n", mentor.into_agent().meta());
    println!("Q: {question}\n");
    world
        .run_with(mentor, Scope::of(Task::new(question)))
        .await??;
    Ok(())
}
