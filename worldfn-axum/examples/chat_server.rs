//! A chat endpoint: question in, progress and answer streamed back over SSE.
//!
//! ```sh
//! cargo run -p worldfn-axum --example chat_server          # fake LLM
//! # then open http://127.0.0.1:3000
//!
//! WORLDFN_PROVIDER=deepseek WORLDFN_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=sk-... \
//!   cargo run -p worldfn-axum --example chat_server --features openai-compat
//! ```
//!
//! Routes:
//! - `GET /`             a tiny HTML page using `EventSource`
//! - `GET /chat?q=...`   runs the agent, streams `status` / `answer` / `failure` events
//! - `GET /agents`       the agent's requirement tree, from its signature
//!
//! The agent knows nothing about axum: it takes an `Input<Question>` and an
//! `Emit<ChatEvent>`; the handler turns the request into a `Scope` and the
//! event stream into an SSE response.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Query, State};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use worldfn::prelude::*;
use worldfn::{AsQuery, LlmError, SseEvent, SseFrame, emit};

#[path = "../../examples/common/mod.rs"]
mod common;

#[derive(Debug, Clone)]
struct Question(String);

/// Lets skills be selected by the question text.
impl AsQuery for Question {
    fn query(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
enum ChatEvent {
    Status(String),
    Answer(String),
    Error(String),
}

impl SseEvent for ChatEvent {
    fn to_sse(&self) -> SseFrame {
        match self {
            ChatEvent::Status(s) => SseFrame::new("status", s.clone()),
            ChatEvent::Answer(s) => SseFrame::new("answer", s.clone()),
            ChatEvent::Error(s) => SseFrame::new("failure", s.clone()),
        }
    }
}

/// Knowledge-base context for the question being asked.
struct Kb {
    passages: Vec<String>,
}

impl ContextSource for Kb {
    type Deps = (Memory, Input<Question>);

    async fn materialize((kb, question): Self::Deps) -> Result<Self, ContextError> {
        Ok(Kb {
            passages: kb.search(&question.0, 3).await?,
        })
    }
}

/// The agent. No web types anywhere in its signature.
async fn support_chat(
    question: Input<Question>,
    llm: Llm,
    kb: Context<Kb>,
    skills: Context<RelevantSkills<2, Question>>,
    events: Emit<ChatEvent>,
) -> Result<String, LlmError> {
    events.send(ChatEvent::Status(format!(
        "found {} relevant passage(s)",
        kb.passages.len()
    )));
    if !skills.skills.is_empty() {
        events.send(ChatEvent::Status(format!(
            "using skills: {}",
            skills.names().join(", ")
        )));
    }
    events.send(ChatEvent::Status("asking the model…".into()));
    let prompt = format!(
        "{}\n\nAnswer using only these passages; say so if they are not enough.\n{}\n\nQuestion: {}",
        skills.prompt(),
        kb.passages
            .iter()
            .map(|p| format!("- {p}"))
            .collect::<Vec<_>>()
            .join("\n"),
        question.0
    );
    let answer = llm.complete(prompt).await?;
    events.send(ChatEvent::Answer(answer.clone()));
    Ok(answer)
}

#[derive(serde::Deserialize)]
struct ChatQuery {
    q: String,
}

async fn chat(State(world): State<Arc<AgentWorld>>, Query(query): Query<ChatQuery>) -> Response {
    let (emitter, events) = emit::channel::<ChatEvent>();
    let run = world.run_with(
        support_chat,
        Scope::of(Question(query.q)).with(emitter.clone()),
    );
    // Report failures in-band: once streaming starts, the status line is sent.
    tokio::spawn(async move {
        let message = match run.await {
            Ok(Ok(_)) => None,
            Ok(Err(llm_error)) => Some(llm_error.to_string()),
            Err(run_error) => Some(run_error.to_string()),
        };
        if let Some(message) = message {
            emitter.send(ChatEvent::Error(message));
        }
        // `emitter` drops here, which ends the SSE response.
    });
    worldfn_axum::sse(events).into_response()
}

async fn agents() -> String {
    support_chat.into_agent().meta().to_string()
}

async fn index() -> Html<&'static str> {
    Html(INDEX)
}

fn knowledge_base() -> FakeMemory {
    FakeMemory::new([
        "worldfn agents are ordinary Rust async functions",
        "worldfn parameter types declare the dependencies an agent needs",
        "worldfn streams events to web clients through Emit and SSE",
        "refunds are processed within five business days",
        "the support team answers tickets on weekdays",
    ])
}

fn app() -> Result<Router, Box<dyn std::error::Error>> {
    let llm = common::llm_from_env("You answer support questions briefly.")?.unwrap_or_else(|| {
        println!("(WORLDFN_PROVIDER unset: using a fake LLM that echoes the passages)");
        Llm::new(FakeLlm::responding(|prompt| {
            let passages: Vec<&str> = prompt.lines().filter(|l| l.starts_with("- ")).collect();
            if passages.is_empty() {
                "I don't know; nothing in the knowledge base matches.".into()
            } else {
                format!("Based on the docs: {}", passages.join(" "))
            }
        }))
    });

    let mut world = AgentWorld::new();
    world
        .provide(llm)?
        .provide_memory(knowledge_base())?
        .provide_skills(SkillLibrary::from_dir(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/skills"
        ))?)?;

    // Fail at startup, not on the first request, if a dependency is missing.
    world.prepare(support_chat)?;
    println!("{}\n", support_chat.into_agent().meta());

    Ok(Router::new()
        .route("/", get(index))
        .route("/chat", get(chat))
        .route("/agents", get(agents))
        .with_state(Arc::new(world)))
}

#[tokio::main]
async fn main() {
    let app = match app() {
        Ok(app) => app,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("port 3000 is free");
    println!("listening on http://127.0.0.1:3000");
    axum::serve(listener, app).await.expect("server runs");
}

const INDEX: &str = r#"<!doctype html>
<meta charset="utf-8">
<title>worldfn chat</title>
<style>
  body { font: 16px system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; padding: 0 1rem; }
  form { display: flex; gap: .5rem; }
  input { flex: 1; padding: .5rem; }
  #log p { margin: .4rem 0; }
  .status { color: #777; font-size: .9em; }
  .error { color: #b00; }
</style>
<h1>worldfn chat</h1>
<form id="f"><input id="q" placeholder="Ask about worldfn or refunds…" autofocus><button>Ask</button></form>
<div id="log"></div>
<script>
const log = document.getElementById("log");
function line(cls, text) {
  const p = document.createElement("p");
  p.className = cls; p.textContent = text; log.append(p);
}
document.getElementById("f").onsubmit = (e) => {
  e.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (!q) return;
  line("question", "› " + q);
  const es = new EventSource("/chat?q=" + encodeURIComponent(q));
  es.addEventListener("status", (ev) => line("status", ev.data));
  es.addEventListener("answer", (ev) => { line("answer", ev.data); es.close(); });
  es.addEventListener("failure", (ev) => { line("error", ev.data); es.close(); });
  // The server closes the stream when the agent finishes; without this the
  // browser would reconnect and ask again.
  es.onerror = () => es.close();
};
</script>
"#;
