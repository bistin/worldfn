use std::borrow::Cow;
use std::fmt;

use crate::param::short_type_name;
use crate::{AgentMeta, Requirement};

/// A runtime-layer failure: the agent could not be prepared or started.
///
/// Kept separate from the function's own output. An agent returning
/// `Result<Answer, AgentError>` runs as
/// `Result<Result<Answer, AgentError>, RunError>`, so callers write `.await??`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// Some declared requirements have no binding in the world.
    Unresolved(Diagnostics),
    /// `start` was called before `initialize`.
    NotPrepared { agent: Cow<'static, str> },
    /// The agent was prepared against a different world.
    ForeignWorld { agent: Cow<'static, str> },
    /// The world's bindings changed since the agent was prepared. Prepare again.
    Stale { agent: Cow<'static, str> },
    /// A parameter failed to resolve for this invocation.
    Param {
        agent: Cow<'static, str>,
        error: ParamError,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Unresolved(diagnostics) => diagnostics.fmt(f),
            RunError::NotPrepared { agent } => {
                write!(f, "agent `{agent}` has not been prepared against a world")
            }
            RunError::ForeignWorld { agent } => {
                write!(f, "agent `{agent}` was prepared against a different world")
            }
            RunError::Stale { agent } => write!(
                f,
                "agent `{agent}` is stale: world bindings changed since it was prepared"
            ),
            RunError::Param { agent, error } => write!(f, "agent `{agent}`: {error}"),
        }
    }
}

impl std::error::Error for RunError {}

impl RunError {
    /// Whether the caller of this invocation is at fault (it left an input or
    /// event sink out of the scope), as opposed to the server's setup or an
    /// upstream dependency. Framework adapters map this to 4xx vs 5xx.
    pub fn is_caller_error(&self) -> bool {
        matches!(
            self,
            RunError::Param { error, .. } if error.kind == ParamErrorKind::MissingFromScope
        )
    }

    /// A suggested HTTP status, for framework adapters:
    /// 400 for caller errors, 502 when a parameter's backend failed
    /// (e.g. retrieval), 500 for setup errors (missing bindings, stale or
    /// unprepared agents).
    pub fn http_status(&self) -> u16 {
        match self {
            _ if self.is_caller_error() => 400,
            RunError::Param { .. } => 502,
            _ => 500,
        }
    }
}

/// Which declared requirements are satisfied, for one agent and one world.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostics {
    pub agent: Cow<'static, str>,
    pub checks: Vec<Check>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// The requirement as declared by the signature.
    pub requirement: Requirement,
    /// `None` if satisfied. Otherwise the unmet part: for a context, only
    /// the needs that are missing.
    pub unmet: Option<Requirement>,
}

impl Check {
    pub fn satisfied(&self) -> bool {
        self.unmet.is_none()
    }
}

impl Diagnostics {
    pub(crate) fn new(meta: &AgentMeta, missing: &[Requirement]) -> Self {
        let checks = meta
            .params
            .iter()
            .map(|requirement| Check {
                requirement: requirement.clone(),
                unmet: missing.iter().find(|m| requirement.same_as(m)).cloned(),
            })
            .collect();
        Self {
            agent: meta.name.clone(),
            checks,
        }
    }

    /// The declared requirements that are not satisfied.
    pub fn missing(&self) -> impl Iterator<Item = &Requirement> {
        self.checks
            .iter()
            .filter(|c| !c.satisfied())
            .map(|c| &c.requirement)
    }
}

/// ```text
/// Cannot prepare researcher:
///   ✓ Llm
///   ✗ Tool<web_search>: no provider registered
///   ✗ Context<RelevantMemory>: needs Memory
/// ```
impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cannot prepare {}:", self.agent)?;
        for check in &self.checks {
            match &check.unmet {
                None if check.requirement.is_per_invocation() => {
                    write!(f, "\n  · {}: checked when started", check.requirement)?
                }
                None => write!(f, "\n  ✓ {}", check.requirement)?,
                Some(Requirement::Context { needs, .. }) => {
                    let needs: Vec<_> = needs.iter().map(ToString::to_string).collect();
                    write!(f, "\n  ✗ {}: needs {}", check.requirement, needs.join(", "))?
                }
                Some(_) => write!(f, "\n  ✗ {}: no provider registered", check.requirement)?,
            }
        }
        Ok(())
    }
}

/// A parameter could not produce its value for one invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamError {
    pub param: &'static str,
    pub kind: ParamErrorKind,
    pub message: String,
}

/// Why a parameter failed to resolve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamErrorKind {
    /// The caller did not supply a per-invocation value (an `Input` or an
    /// `Emit` sink) in the `Scope`.
    MissingFromScope,
    /// Resolution itself failed, e.g. a retrieval backend errored.
    Failed,
}

impl ParamError {
    pub fn missing_from_scope(param: &'static str) -> Self {
        Self {
            param,
            kind: ParamErrorKind::MissingFromScope,
            message: "not present in the invocation scope".into(),
        }
    }

    pub fn failed(param: &'static str, message: impl Into<String>) -> Self {
        Self {
            param,
            kind: ParamErrorKind::Failed,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parameter `{}` failed to resolve: {}",
            short_type_name(self.param),
            self.message
        )
    }
}

impl std::error::Error for ParamError {}

/// A binding already exists for this key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindError {
    pub type_name: &'static str,
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "`{}` is already bound in this world; use `replace` to rebind",
            self.type_name
        )
    }
}

impl std::error::Error for BindError {}
