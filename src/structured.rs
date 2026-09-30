//! Typed replies: `llm.complete_as::<T>(request)`.
//!
//! The JSON Schema is generated from `T` (via `schemars`) and sent as the
//! request's output format; providers use native structured output where they
//! have it. The reply is then *validated by deserializing it into `T`*, which
//! is the check that matters to the caller. If that fails, the model is shown
//! its reply and the error and asked again, up to a retry limit.

use std::fmt;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::chat::{ChatRequest, FinishReason, Message, OutputFormat};
use crate::{Llm, LlmError};

/// A typed reply could not be obtained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredError {
    Llm(LlmError),
    /// Every attempt produced output that did not deserialize into `T`.
    Invalid {
        reason: String,
        /// The last reply, verbatim.
        raw: String,
        attempts: u32,
    },
}

impl fmt::Display for StructuredError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructuredError::Llm(e) => e.fmt(f),
            StructuredError::Invalid {
                reason, attempts, ..
            } => write!(
                f,
                "model output did not match the expected type after {attempts} attempt(s): {reason}"
            ),
        }
    }
}

impl std::error::Error for StructuredError {}

impl From<LlmError> for StructuredError {
    fn from(e: LlmError) -> Self {
        StructuredError::Llm(e)
    }
}

/// The output format for `T`: its schema name and JSON Schema.
pub fn output_format_for<T: JsonSchema>() -> OutputFormat {
    let schema = schemars::schema_for!(T);
    OutputFormat::Json {
        name: T::schema_name().into_owned(),
        schema: serde_json::to_string(&schema).expect("a generated schema serializes"),
    }
}

/// Models often wrap JSON in a Markdown fence; accept that, nothing looser.
fn strip_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

impl Llm {
    /// Ask for a reply of type `T`, retrying once on invalid output.
    pub async fn complete_as<T>(&self, request: ChatRequest) -> Result<T, StructuredError>
    where
        T: DeserializeOwned + JsonSchema,
    {
        self.complete_as_retrying(request, 1).await
    }

    /// Ask for a reply of type `T`, allowing `retries` corrective retries.
    pub async fn complete_as_retrying<T>(
        &self,
        request: ChatRequest,
        retries: u32,
    ) -> Result<T, StructuredError>
    where
        T: DeserializeOwned + JsonSchema,
    {
        let mut request = request.output(output_format_for::<T>());
        let mut attempts = 0;
        loop {
            attempts += 1;
            let response = self.chat(request.clone()).await?;
            let raw = response.message.text();
            let parsed = if response.finish == FinishReason::Length {
                Err("the reply was cut off by the output token limit".to_owned())
            } else {
                serde_json::from_str::<T>(strip_fence(&raw)).map_err(|e| e.to_string())
            };
            match parsed {
                Ok(value) => return Ok(value),
                Err(reason) if attempts > retries => {
                    return Err(StructuredError::Invalid {
                        reason,
                        raw,
                        attempts,
                    });
                }
                Err(reason) => {
                    request = request.message(Message::assistant(raw)).user(format!(
                        "That reply was not valid: {reason}. Reply again with only \
                             the JSON value, matching the schema exactly."
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::strip_fence;

    #[test]
    fn fences_are_stripped_but_nothing_else() {
        assert_eq!(strip_fence("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_fence("```\n[1]\n```"), "[1]");
        assert_eq!(strip_fence("  {\"a\":1} "), "{\"a\":1}");
        assert_eq!(strip_fence("Sure! {\"a\":1}"), "Sure! {\"a\":1}");
    }
}
