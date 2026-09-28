//! Memory parameters scoped to the caller.
//!
//! The web layer (or any caller) authenticates the request and puts a
//! [`Principal`] into the invocation `Scope`. Every parameter here binds to
//! that principal while it is resolved, so an agent can only ever reach the
//! caller's own account and session. There is no parameter that takes an
//! account id from the agent.
//!
//! | Parameter | Access |
//! |---|---|
//! | `Context<Conversation<N>>` | read: the last `N` turns of this session |
//! | `Context<Recall<N, Q>>` | read: up to `N` long-term memories relevant to input `Q` |
//! | [`SessionLog`] | read + append, this session |
//! | [`AccountMemory`] | read + remember, this account |
//!
//! Declaring only the `Context` types makes an agent read-only by
//! construction; the handles are the write capability.

use std::any::type_name;
use std::fmt;
use std::future::{Ready, ready};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::store::{
    AccountId, AccountMemoryStore, MemoryEntry, Role, SessionId, SessionStore, StoreError, Turn,
};
use crate::{
    AgentParam, AgentWorld, AsQuery, ContextError, ContextSource, Input, ParamError, Requirement,
    Scope, Task,
};

/// Who this invocation is for. Put it in the `Scope` after authenticating;
/// worldfn trusts it as given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    pub account: AccountId,
    pub session: SessionId,
}

impl Principal {
    pub fn new(account: impl Into<String>, session: impl Into<String>) -> Self {
        Self {
            account: AccountId::new(account),
            session: SessionId::new(session),
        }
    }
}

/// World binding for the session store (see `AgentWorld::provide_sessions`).
#[derive(Clone)]
pub(crate) struct Sessions(pub Arc<dyn SessionStore>);

/// World binding for account memory (see `AgentWorld::provide_account_memory`).
#[derive(Clone)]
pub(crate) struct AccountMemories(pub Arc<dyn AccountMemoryStore>);

fn principal(scope: &Scope, param: &'static str) -> Result<Principal, ParamError> {
    scope
        .get::<Principal>()
        .map(|p| (*p).clone())
        .ok_or(ParamError {
            param,
            kind: crate::ParamErrorKind::MissingFromScope,
            message: "no Principal in the invocation scope".into(),
        })
}

fn principal_requirement() -> Requirement {
    Requirement::Input {
        type_name: type_name::<Principal>(),
    }
}

impl From<StoreError> for ContextError {
    fn from(e: StoreError) -> Self {
        ContextError(e.0)
    }
}

/// Parameter: this session's conversation log (read + append).
#[derive(Clone)]
pub struct SessionLog {
    store: Arc<dyn SessionStore>,
    principal: Principal,
}

impl SessionLog {
    /// The last `n` turns, oldest first.
    pub async fn recent(&self, n: usize) -> Result<Vec<Turn>, StoreError> {
        self.store
            .recent(&self.principal.account, &self.principal.session, n)
            .await
    }

    pub async fn append(&self, turn: Turn) -> Result<(), StoreError> {
        self.store
            .append(&self.principal.account, &self.principal.session, turn)
            .await
    }

    /// Append a user turn and the assistant's reply.
    pub async fn record_exchange(
        &self,
        question: impl Into<String>,
        answer: impl Into<String>,
    ) -> Result<(), StoreError> {
        self.append(Turn::user(question)).await?;
        self.append(Turn::assistant(answer)).await
    }

    pub fn principal(&self) -> &Principal {
        &self.principal
    }
}

impl fmt::Debug for SessionLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SessionLog")
            .field("principal", &self.principal)
            .finish()
    }
}

impl AgentParam for SessionLog {
    type State = Arc<dyn SessionStore>;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Service {
            type_name: "SessionStore",
        });
        out.push(principal_requirement());
    }

    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
        world
            .resource::<Sessions>()
            .map(|s| s.0.clone())
            .ok_or_else(|| {
                vec![Requirement::Service {
                    type_name: "SessionStore",
                }]
            })
    }

    fn resolve(state: &mut Self::State, _world: &AgentWorld, scope: &Scope) -> Self::Future {
        ready(
            principal(scope, type_name::<Self>()).map(|principal| SessionLog {
                store: state.clone(),
                principal,
            }),
        )
    }
}

/// Parameter: this account's long-term memory (search + remember).
#[derive(Clone)]
pub struct AccountMemory {
    store: Arc<dyn AccountMemoryStore>,
    account: AccountId,
}

impl AccountMemory {
    pub async fn search(&self, query: &str, n: usize) -> Result<Vec<MemoryEntry>, StoreError> {
        self.store.search(&self.account, query, n).await
    }

    pub async fn remember(&self, text: impl Into<String>) -> Result<(), StoreError> {
        self.store
            .remember(&self.account, MemoryEntry::new(text))
            .await
    }

    pub fn account(&self) -> &AccountId {
        &self.account
    }
}

impl fmt::Debug for AccountMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccountMemory")
            .field("account", &self.account)
            .finish()
    }
}

impl AgentParam for AccountMemory {
    type State = Arc<dyn AccountMemoryStore>;
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Service {
            type_name: "AccountMemoryStore",
        });
        out.push(principal_requirement());
    }

    fn init(world: &AgentWorld) -> Result<Self::State, Vec<Requirement>> {
        world
            .resource::<AccountMemories>()
            .map(|s| s.0.clone())
            .ok_or_else(|| {
                vec![Requirement::Service {
                    type_name: "AccountMemoryStore",
                }]
            })
    }

    fn resolve(state: &mut Self::State, _world: &AgentWorld, scope: &Scope) -> Self::Future {
        ready(
            principal(scope, type_name::<Self>()).map(|p| AccountMemory {
                store: state.clone(),
                account: p.account,
            }),
        )
    }
}

/// Context: the last `N` turns of the caller's session, oldest first.
#[derive(Debug, Clone)]
pub struct Conversation<const N: usize> {
    pub turns: Vec<Turn>,
}

impl<const N: usize> Conversation<N> {
    /// `user: …` / `assistant: …` lines, for a prompt.
    pub fn transcript(&self) -> String {
        self.turns
            .iter()
            .map(|t| {
                let who = match t.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                };
                format!("{who}: {}", t.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl<const N: usize> ContextSource for Conversation<N> {
    type Deps = SessionLog;

    async fn materialize(log: SessionLog) -> Result<Self, ContextError> {
        Ok(Conversation {
            turns: log.recent(N).await?,
        })
    }
}

/// Context: up to `N` of the caller's long-term memories relevant to this
/// invocation's input `Q` (by default the `Task`).
pub struct Recall<const N: usize = 5, Q = Task> {
    pub entries: Vec<MemoryEntry>,
    _query: PhantomData<fn() -> Q>,
}

impl<const N: usize, Q> Recall<N, Q> {
    pub fn texts(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.text.as_str()).collect()
    }
}

impl<const N: usize, Q> fmt::Debug for Recall<N, Q> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Recall")
            .field("entries", &self.entries)
            .finish()
    }
}

impl<const N: usize, Q: AsQuery> ContextSource for Recall<N, Q> {
    type Deps = (AccountMemory, Input<Q>);

    async fn materialize((memory, input): Self::Deps) -> Result<Self, ContextError> {
        Ok(Recall {
            entries: memory.search(input.query(), N).await?,
            _query: PhantomData,
        })
    }
}
