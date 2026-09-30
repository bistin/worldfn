//! Support-ticket triage: an LLM step inside a backend, with typed output.
//!
//! ```sh
//! cargo run --example triage                       # fake LLM, no keys needed
//! cargo test --example triage                      # the same agent under test
//!
//! WORLDFN_PROVIDER=deepseek WORLDFN_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=sk-... \
//!   cargo run --example triage --features openai-compat
//! WORLDFN_PROVIDER=codex WORLDFN_MODEL=gpt-5.5 \
//!   cargo run --example triage --features codex
//! ```
//!
//! What it shows:
//! - the ticket is a per-invocation `Input`, not global state;
//! - `Context<SimilarTickets<3>>` retrieves past resolutions *for this ticket*
//!   before the body runs;
//! - the model's text becomes a typed `Triage`, and malformed output is a
//!   domain error the caller handles, not a panic or a runtime failure;
//! - one prepared agent, many tickets, all in flight at once.

use std::fmt;

use worldfn::prelude::*;
use worldfn::{ChatRequest, StructuredError};

mod common;

#[derive(Debug, Clone)]
pub struct Ticket {
    pub id: u32,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Billing,
    Bug,
    Account,
    FeatureRequest,
    Other,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    Low,
    Normal,
    Urgent,
}

/// The typed reply. Its JSON Schema is generated from this type and sent to
/// the model; the reply is validated by deserializing into it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, schemars::JsonSchema)]
pub struct Triage {
    pub category: Category,
    pub priority: Priority,
    /// A short first reply to the customer.
    pub reply: String,
}

#[derive(Debug, PartialEq)]
pub enum TriageError {
    Llm(worldfn::LlmError),
    /// The model answered, but not in the agreed shape.
    BadModelOutput {
        reason: String,
        raw: String,
    },
}

impl fmt::Display for TriageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TriageError::Llm(e) => write!(f, "{e}"),
            TriageError::BadModelOutput { reason, .. } => write!(f, "bad model output: {reason}"),
        }
    }
}

/// Context: up to `N` past tickets, with their resolutions, similar to the
/// ticket being triaged.
pub struct SimilarTickets<const N: usize> {
    pub examples: Vec<String>,
}

impl<const N: usize> ContextSource for SimilarTickets<N> {
    type Deps = (Memory, Input<Ticket>);

    async fn materialize((history, ticket): Self::Deps) -> Result<Self, ContextError> {
        Ok(SimilarTickets {
            examples: history.search(&ticket.text, N).await?,
        })
    }
}

/// The agent. Its signature is the whole integration contract.
pub async fn triage(
    ticket: Input<Ticket>,
    llm: Llm,
    similar: Context<SimilarTickets<3>>,
) -> Result<Triage, TriageError> {
    let examples = if similar.examples.is_empty() {
        "(none)".to_owned()
    } else {
        similar
            .examples
            .iter()
            .map(|e| format!("- {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let request = ChatRequest::new()
        .system("You triage customer support tickets.")
        .user(format!(
            "Similar past tickets and how they were resolved:\n{examples}\n\n\
             Ticket #{}:\n{}",
            ticket.id, ticket.text
        ));
    // One corrective retry if the reply does not fit `Triage`.
    let triage: Triage = llm
        .complete_as_retrying(request, 1)
        .await
        .map_err(|e| match e {
            StructuredError::Llm(e) => TriageError::Llm(e),
            StructuredError::Invalid { reason, raw, .. } => {
                TriageError::BadModelOutput { reason, raw }
            }
        })?;
    // Rules the type cannot express are still checked in code.
    if triage.reply.trim().is_empty() {
        return Err(TriageError::BadModelOutput {
            reason: "empty reply".into(),
            raw: String::new(),
        });
    }
    Ok(triage)
}

fn history() -> FakeMemory {
    FakeMemory::new([
        "billing: charged twice for the monthly plan -> refunded the duplicate charge",
        "billing: invoice shows the wrong company name -> reissued invoice",
        "bug: export to CSV crashes with an error -> fixed in 2.3.1, asked user to update",
        "account: cannot log in after password reset -> cleared stale session",
        "feature_request: dark mode request -> added to roadmap, thanked user",
    ])
}

/// A stand-in model that answers in the expected JSON by keyword, so the
/// example runs without credentials.
fn keyword_llm() -> FakeLlm {
    FakeLlm::responding(|prompt| {
        let ticket = prompt
            .split("Ticket #")
            .nth(1)
            .unwrap_or_default()
            .to_lowercase();
        let (category, priority) = if ticket.contains("charged") || ticket.contains("invoice") {
            ("billing", "normal")
        } else if ticket.contains("crash") || ticket.contains("error") {
            ("bug", "urgent")
        } else if ticket.contains("log in") || ticket.contains("password") {
            ("account", "urgent")
        } else {
            ("other", "low")
        };
        format!(
            "{{\"category\":\"{category}\",\"priority\":\"{priority}\",\
             \"reply\":\"Thanks for reaching out, we're looking into it.\"}}"
        )
    })
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let llm = match common::llm_from_env("You triage customer support tickets.")? {
        Some(llm) => llm,
        None => {
            println!("(WORLDFN_PROVIDER unset: using a keyword-based fake LLM)\n");
            Llm::new(keyword_llm())
        }
    };

    let mut world = AgentWorld::new();
    world.provide(llm)?.provide_memory(history())?;
    println!("{}\n", triage.into_agent().meta());

    let tickets = [
        "I was charged twice this month, please fix!",
        "The app crashes with an error every time I export to CSV.",
        "Can't log in since I reset my password yesterday.",
        "Would love a keyboard shortcut for search.",
    ];

    // Prepare once, then start every ticket; the run futures are 'static, so
    // they can all be spawned and resolved concurrently.
    let mut agent = world.prepare(triage)?;
    let handles: Vec<_> = tickets
        .iter()
        .zip(1..)
        .map(|(text, id)| {
            let ticket = Ticket {
                id,
                text: text.to_string(),
            };
            tokio::spawn(world.run_prepared_with(&mut agent, Scope::of(ticket)))
        })
        .collect();

    for (handle, text) in handles.into_iter().zip(tickets) {
        match handle.await?? {
            Ok(t) => println!(
                "{:<16} {:<7} {text}\n{:>25}↳ {}",
                format!("{:?}", t.category),
                format!("{:?}", t.priority),
                "",
                t.reply
            ),
            Err(e) => println!("{:<24} {text}\n{:>25}↳ {e}", "NEEDS HUMAN", ""),
        }
    }
    Ok(())
}

/// The same agent under test: only the world changes.
#[cfg(test)]
mod tests {
    use super::*;

    fn world(llm: FakeLlm) -> AgentWorld {
        let mut world = AgentWorld::new();
        world
            .provide_llm(llm)
            .unwrap()
            .provide_memory(history())
            .unwrap();
        world
    }

    fn ticket(text: &str) -> Scope {
        Scope::of(Ticket {
            id: 7,
            text: text.into(),
        })
    }

    #[tokio::test]
    async fn parses_model_json_into_typed_triage() {
        let llm = FakeLlm::with_answer(
            "```json\n{\"category\":\"billing\",\"priority\":\"urgent\",\"reply\":\"Refunding now.\"}\n```",
        );
        let result = world(llm.clone())
            .run_with(triage, ticket("charged twice"))
            .await
            .unwrap();
        assert_eq!(
            result,
            Ok(Triage {
                category: Category::Billing,
                priority: Priority::Urgent,
                reply: "Refunding now.".into(),
            })
        );
        // The prompt carried the relevant history, not the unrelated entries.
        let prompt = &llm.prompts()[0];
        assert!(prompt.contains("refunded the duplicate charge"), "{prompt}");
        assert!(!prompt.contains("dark mode"), "{prompt}");
    }

    #[tokio::test]
    async fn malformed_output_gets_one_correction_then_is_a_domain_error() {
        // First reply is prose, the correction is well-formed: success.
        let llm = FakeLlm::new()
            .then_answer("Sure! It's a billing issue.")
            .then_answer(r#"{"category":"billing","priority":"low","reply":"On it."}"#);
        let result = world(llm.clone())
            .run_with(triage, ticket("charged twice"))
            .await
            .unwrap();
        assert_eq!(result.map(|t| t.category), Ok(Category::Billing));
        assert_eq!(llm.calls(), 2);

        // Two bad replies: the caller gets the last one and the reason.
        let last = r#"{"category":"refund","priority":"low","reply":"x"}"#;
        let result = world(FakeLlm::scripted(["not json", last]))
            .run_with(triage, ticket("anything"))
            .await
            .unwrap();
        match result {
            Err(TriageError::BadModelOutput { reason, raw }) => {
                assert_eq!(raw, last);
                assert!(reason.contains("unknown variant `refund`"), "{reason}");
            }
            other => panic!("expected BadModelOutput, got {other:?}"),
        }

        // Well-formed but empty reply: caught by the check after parsing.
        let result = world(FakeLlm::with_answer(
            r#"{"category":"bug","priority":"low","reply":" "}"#,
        ))
        .run_with(triage, ticket("anything"))
        .await
        .unwrap();
        assert!(
            matches!(result, Err(TriageError::BadModelOutput { reason, .. }) if reason == "empty reply")
        );
    }

    #[tokio::test]
    async fn provider_failure_reaches_the_caller_as_a_value() {
        let llm = FakeLlm::new().then_error("rate limited");
        let result = world(llm).run_with(triage, ticket("x")).await.unwrap();
        assert_eq!(
            result,
            Err(TriageError::Llm(worldfn::LlmError("rate limited".into())))
        );
    }

    #[tokio::test]
    async fn keyword_fake_routes_the_demo_tickets() {
        let world = world(keyword_llm());
        let t = world
            .run_with(triage, ticket("The app crashes on export"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!((t.category, t.priority), (Category::Bug, Priority::Urgent));
    }
}
