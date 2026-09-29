//! Per-account and per-session memory storage, behind swappable traits.
//!
//! Two kinds of memory, two traits:
//!
//! | | [`SessionStore`] | [`AccountMemoryStore`] |
//! |---|---|---|
//! | holds | the turns of one conversation | long-term facts about one account |
//! | read by | recency (last `n` turns) | relevance to a query |
//! | typical backend | Postgres, cached in Redis | Postgres (+ pgvector) |
//!
//! Every method takes the owning [`AccountId`] (and [`SessionId`]), so no
//! backend can serve data without knowing whose it is, and a session id
//! guessed from another account reads nothing.
//!
//! [`Cached`] puts any [`Cache`] (Redis, or [`InMemoryCache`] in tests) in
//! front of a store; cache logic lives here once, so a cache backend only
//! implements get/set/delete. [`conformance`] is the shared test suite every
//! store and every cached combination must pass.
//!
//! Agents never see these traits. They see scoped handles such as
//! `SessionLog` and `AccountMemory`, bound to the caller's `Principal`.

use std::collections::HashMap;
use std::fmt;
use std::future::ready;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::BoxFuture;
use crate::memory::words;

/// A storage operation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreError(pub String);

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "store error: {}", self.0)
    }
}

impl std::error::Error for StoreError {}

macro_rules! id_type {
    ($name:ident, $what:literal) => {
        #[doc = concat!("Identifies ", $what, ". Opaque to worldfn.")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

id_type!(AccountId, "an account (a user or tenant)");
id_type!(SessionId, "one conversation of an account");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    User,
    Assistant,
}

/// One message in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Turn {
    pub role: Role,
    pub text: String,
}

impl Turn {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            text: text.into(),
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            text: text.into(),
        }
    }
}

/// One long-term memory of an account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEntry {
    pub text: String,
}

impl MemoryEntry {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Conversation history, keyed by account *and* session.
pub trait SessionStore: Send + Sync + 'static {
    fn append<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
        turn: Turn,
    ) -> BoxFuture<'a, Result<(), StoreError>>;

    /// The last `n` turns, oldest first.
    fn recent<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<Turn>, StoreError>>;

    fn clear<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
    ) -> BoxFuture<'a, Result<(), StoreError>>;

    /// Delete every session of `account`.
    fn forget_account<'a>(
        &'a self,
        account: &'a AccountId,
    ) -> BoxFuture<'a, Result<(), StoreError>>;
}

/// Long-term memory, keyed by account.
pub trait AccountMemoryStore: Send + Sync + 'static {
    fn remember<'a>(
        &'a self,
        account: &'a AccountId,
        entry: MemoryEntry,
    ) -> BoxFuture<'a, Result<(), StoreError>>;

    /// Up to `n` entries relevant to `query`, best first.
    fn search<'a>(
        &'a self,
        account: &'a AccountId,
        query: &'a str,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<MemoryEntry>, StoreError>>;

    /// Delete everything remembered about `account`.
    fn forget_account<'a>(
        &'a self,
        account: &'a AccountId,
    ) -> BoxFuture<'a, Result<(), StoreError>>;
}

/// A string key-value cache with expiry: all a cache backend (e.g. Redis)
/// has to implement to sit in front of a store via [`Cached`].
pub trait Cache: Send + Sync + 'static {
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<String>, StoreError>>;

    fn set<'a>(
        &'a self,
        key: &'a str,
        value: String,
        ttl: Duration,
    ) -> BoxFuture<'a, Result<(), StoreError>>;

    fn delete<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<(), StoreError>>;

    /// Delete every key starting with `prefix`.
    fn delete_prefix<'a>(&'a self, prefix: &'a str) -> BoxFuture<'a, Result<(), StoreError>>;
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

fn done<'a, T: Send + 'a>(value: Result<T, StoreError>) -> BoxFuture<'a, Result<T, StoreError>> {
    Box::pin(ready(value))
}

type SessionMap = HashMap<(AccountId, SessionId), Vec<Turn>>;

/// Sessions in process memory. Clones share state.
#[derive(Clone, Default)]
pub struct InMemorySessions {
    turns: Arc<Mutex<SessionMap>>,
}

impl SessionStore for InMemorySessions {
    fn append<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
        turn: Turn,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        lock(&self.turns)
            .entry((account.clone(), session.clone()))
            .or_default()
            .push(turn);
        done(Ok(()))
    }

    fn recent<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<Turn>, StoreError>> {
        let turns = lock(&self.turns);
        let all = turns
            .get(&(account.clone(), session.clone()))
            .map(Vec::as_slice)
            .unwrap_or_default();
        done(Ok(all[all.len().saturating_sub(n)..].to_vec()))
    }

    fn clear<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        lock(&self.turns).remove(&(account.clone(), session.clone()));
        done(Ok(()))
    }

    fn forget_account<'a>(
        &'a self,
        account: &'a AccountId,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        lock(&self.turns).retain(|(a, _), _| a != account);
        done(Ok(()))
    }
}

/// Long-term memory in process memory, ranked by word overlap with the
/// query (a deterministic stand-in for vector search). Clones share state.
#[derive(Clone, Default)]
pub struct InMemoryAccountMemory {
    entries: Arc<Mutex<HashMap<AccountId, Vec<MemoryEntry>>>>,
}

impl AccountMemoryStore for InMemoryAccountMemory {
    fn remember<'a>(
        &'a self,
        account: &'a AccountId,
        entry: MemoryEntry,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        lock(&self.entries)
            .entry(account.clone())
            .or_default()
            .push(entry);
        done(Ok(()))
    }

    fn search<'a>(
        &'a self,
        account: &'a AccountId,
        query: &'a str,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<MemoryEntry>, StoreError>> {
        let entries = lock(&self.entries);
        let query = words(query);
        let mut scored: Vec<(usize, &MemoryEntry)> = entries
            .get(account)
            .into_iter()
            .flatten()
            .map(|e| (words(&e.text).intersection(&query).count(), e))
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        done(Ok(scored
            .into_iter()
            .take(n)
            .map(|(_, e)| e.clone())
            .collect()))
    }

    fn forget_account<'a>(
        &'a self,
        account: &'a AccountId,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        lock(&self.entries).remove(account);
        done(Ok(()))
    }
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<String, (String, Instant)>,
    down: bool,
    gets: usize,
    hits: usize,
}

/// A [`Cache`] in process memory, honoring TTLs. Can be switched "down" to
/// test that callers survive a cache outage. Clones share state.
#[derive(Clone, Default)]
pub struct InMemoryCache {
    state: Arc<Mutex<CacheState>>,
}

impl InMemoryCache {
    /// Simulate an outage: every operation fails until turned back on.
    pub fn set_down(&self, down: bool) {
        lock(&self.state).down = down;
    }

    /// `(gets, hits)` so far.
    pub fn stats(&self) -> (usize, usize) {
        let state = lock(&self.state);
        (state.gets, state.hits)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        lock(&self.state).entries.contains_key(key)
    }

    fn check(state: &CacheState) -> Result<(), StoreError> {
        if state.down {
            Err(StoreError("cache unavailable".into()))
        } else {
            Ok(())
        }
    }
}

impl Cache for InMemoryCache {
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<String>, StoreError>> {
        let mut state = lock(&self.state);
        let result = Self::check(&state).map(|()| {
            state.gets += 1;
            let now = Instant::now();
            state.entries.retain(|_, (_, expires)| *expires > now);
            let value = state.entries.get(key).map(|(v, _)| v.clone());
            state.hits += usize::from(value.is_some());
            value
        });
        done(result)
    }

    fn set<'a>(
        &'a self,
        key: &'a str,
        value: String,
        ttl: Duration,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        let mut state = lock(&self.state);
        let result = Self::check(&state).map(|()| {
            state
                .entries
                .insert(key.to_owned(), (value, Instant::now() + ttl));
        });
        done(result)
    }

    fn delete<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<(), StoreError>> {
        let mut state = lock(&self.state);
        let result = Self::check(&state).map(|()| {
            state.entries.remove(key);
        });
        done(result)
    }

    fn delete_prefix<'a>(&'a self, prefix: &'a str) -> BoxFuture<'a, Result<(), StoreError>> {
        let mut state = lock(&self.state);
        let result = Self::check(&state).map(|()| {
            state.entries.retain(|k, _| !k.starts_with(prefix));
        });
        done(result)
    }
}

/// A store with a cache in front of it. The wrapped store stays the source of
/// truth; the cache only makes reads cheaper.
///
/// Sessions:
/// - `recent(n)` for `n <= window` is served from the cache, filled from the
///   store on a miss with the last `window` turns;
/// - writes go to the store first, then invalidate the session's cache entry;
/// - if the cache is unavailable, reads fall back to the store. If an
///   invalidation fails, the stale entry lives at most `ttl`.
///
/// Account memory: searches go to the store (query-dependent results cache
/// poorly). `forget_account` purges the store *and* every cached entry of the
/// account, and fails if the purge cannot be confirmed: deletion is not
/// reported done while copies may remain.
pub struct Cached<S, C> {
    store: S,
    cache: C,
    window: usize,
    ttl: Duration,
}

impl<S, C> Cached<S, C> {
    /// Defaults: a 50-turn window, 10-minute TTL.
    pub fn new(store: S, cache: C) -> Self {
        Self {
            store,
            cache,
            window: 50,
            ttl: Duration::from_secs(600),
        }
    }

    /// How many recent turns per session the cache holds.
    pub fn window(mut self, turns: usize) -> Self {
        self.window = turns;
        self
    }

    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }
}

/// Cache keys length-prefix every id, so `("a:b", "c")` and `("a", "b:c")`
/// cannot collide and an account's prefix never matches another account.
fn account_prefix(account: &AccountId) -> String {
    format!("worldfn:session:{}:{}:", account.0.len(), account.0)
}

fn session_key(account: &AccountId, session: &SessionId) -> String {
    format!(
        "{}{}:{}",
        account_prefix(account),
        session.0.len(),
        session.0
    )
}

/// `role:text` lines with `\` and newlines escaped.
fn encode_turns(turns: &[Turn]) -> String {
    turns
        .iter()
        .map(|t| {
            let role = match t.role {
                Role::User => 'u',
                Role::Assistant => 'a',
            };
            let text = t.text.replace('\\', "\\\\").replace('\n', "\\n");
            format!("{role}:{text}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn decode_turns(encoded: &str) -> Option<Vec<Turn>> {
    if encoded.is_empty() {
        return Some(Vec::new());
    }
    encoded
        .split('\n')
        .map(|line| {
            let (role, text) = line.split_once(':')?;
            let role = match role {
                "u" => Role::User,
                "a" => Role::Assistant,
                _ => return None,
            };
            let mut out = String::new();
            let mut chars = text.chars();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    match chars.next()? {
                        'n' => out.push('\n'),
                        '\\' => out.push('\\'),
                        _ => return None,
                    }
                } else {
                    out.push(c);
                }
            }
            Some(Turn { role, text: out })
        })
        .collect()
}

impl<S: SessionStore, C: Cache> SessionStore for Cached<S, C> {
    fn append<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
        turn: Turn,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async move {
            self.store.append(account, session, turn).await?;
            // Stale for at most `ttl` if this fails; the write itself succeeded.
            let _ = self.cache.delete(&session_key(account, session)).await;
            Ok(())
        })
    }

    fn recent<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<Turn>, StoreError>> {
        Box::pin(async move {
            if n > self.window {
                return self.store.recent(account, session, n).await;
            }
            let key = session_key(account, session);
            if let Ok(Some(hit)) = self.cache.get(&key).await {
                if let Some(turns) = decode_turns(&hit) {
                    return Ok(turns[turns.len().saturating_sub(n)..].to_vec());
                }
            }
            let turns = self.store.recent(account, session, self.window).await?;
            let _ = self.cache.set(&key, encode_turns(&turns), self.ttl).await;
            Ok(turns[turns.len().saturating_sub(n)..].to_vec())
        })
    }

    fn clear<'a>(
        &'a self,
        account: &'a AccountId,
        session: &'a SessionId,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async move {
            self.store.clear(account, session).await?;
            self.cache.delete(&session_key(account, session)).await
        })
    }

    fn forget_account<'a>(
        &'a self,
        account: &'a AccountId,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async move {
            SessionStore::forget_account(&self.store, account).await?;
            self.cache.delete_prefix(&account_prefix(account)).await
        })
    }
}

impl<S: AccountMemoryStore, C: Cache> AccountMemoryStore for Cached<S, C> {
    fn remember<'a>(
        &'a self,
        account: &'a AccountId,
        entry: MemoryEntry,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        self.store.remember(account, entry)
    }

    fn search<'a>(
        &'a self,
        account: &'a AccountId,
        query: &'a str,
        n: usize,
    ) -> BoxFuture<'a, Result<Vec<MemoryEntry>, StoreError>> {
        self.store.search(account, query, n)
    }

    fn forget_account<'a>(
        &'a self,
        account: &'a AccountId,
    ) -> BoxFuture<'a, Result<(), StoreError>> {
        Box::pin(async move {
            AccountMemoryStore::forget_account(&self.store, account).await?;
            self.cache.delete_prefix(&account_prefix(account)).await
        })
    }
}

/// The shared behavior every store must have. Backend crates call these from
/// their own tests with a factory for a fresh, empty store; each function
/// panics with a description of the first violated rule.
pub mod conformance {
    use super::*;

    fn ids() -> (AccountId, AccountId, SessionId, SessionId) {
        (
            AccountId::new("alice"),
            AccountId::new("bob"),
            SessionId::new("s1"),
            SessionId::new("s2"),
        )
    }

    /// Ordering, windowing, isolation between accounts and sessions, clear,
    /// and account deletion.
    pub async fn session_store<S: SessionStore>(store: S) {
        let (alice, bob, s1, s2) = ids();
        assert!(
            store.recent(&alice, &s1, 10).await.unwrap().is_empty(),
            "a new session is empty"
        );
        for i in 0..5 {
            store
                .append(&alice, &s1, Turn::user(format!("q{i}")))
                .await
                .unwrap();
            store
                .append(&alice, &s1, Turn::assistant(format!("a{i}\nline 2")))
                .await
                .unwrap();
        }
        let last3 = store.recent(&alice, &s1, 3).await.unwrap();
        assert_eq!(
            last3,
            [
                Turn::assistant("a3\nline 2"),
                Turn::user("q4"),
                Turn::assistant("a4\nline 2")
            ],
            "recent(n) returns the last n turns, oldest first, text intact"
        );
        assert_eq!(
            store.recent(&alice, &s1, 100).await.unwrap().len(),
            10,
            "recent(n) with n beyond the history returns all of it"
        );

        store
            .append(&alice, &s2, Turn::user("other"))
            .await
            .unwrap();
        store.append(&bob, &s1, Turn::user("bob's")).await.unwrap();
        assert_eq!(
            store.recent(&bob, &s1, 100).await.unwrap(),
            [Turn::user("bob's")],
            "the same session id under another account is a different session"
        );
        assert_eq!(
            store.recent(&alice, &s2, 100).await.unwrap(),
            [Turn::user("other")],
            "sessions of one account are separate"
        );

        store.clear(&alice, &s2).await.unwrap();
        assert!(
            store.recent(&alice, &s2, 10).await.unwrap().is_empty(),
            "clear empties the session"
        );
        assert_eq!(
            store.recent(&alice, &s1, 100).await.unwrap().len(),
            10,
            "clear leaves other sessions alone"
        );

        SessionStore::forget_account(&store, &alice).await.unwrap();
        assert!(
            store.recent(&alice, &s1, 100).await.unwrap().is_empty(),
            "forget_account deletes the account's sessions"
        );
        assert_eq!(
            store.recent(&bob, &s1, 100).await.unwrap().len(),
            1,
            "forget_account leaves other accounts alone"
        );
    }

    /// Relevance, limits, isolation between accounts, and account deletion.
    pub async fn account_memory_store<S: AccountMemoryStore>(store: S) {
        let (alice, bob, _, _) = ids();
        for fact in [
            "prefers tokio for async rust",
            "deploys to kubernetes on fridays",
            "has a cat called Ferris",
        ] {
            store
                .remember(&alice, MemoryEntry::new(fact))
                .await
                .unwrap();
        }
        store
            .remember(&bob, MemoryEntry::new("prefers async-std for async rust"))
            .await
            .unwrap();

        let hits = store
            .search(&alice, "which async rust runtime", 5)
            .await
            .unwrap();
        assert_eq!(
            hits.first().map(|e| e.text.as_str()),
            Some("prefers tokio for async rust"),
            "search ranks the relevant entry first"
        );
        assert!(
            hits.iter().all(|e| !e.text.contains("async-std")),
            "search never returns another account's entries"
        );
        assert!(
            store
                .search(&alice, "rust kubernetes cat", 1)
                .await
                .unwrap()
                .len()
                <= 1,
            "search respects the limit"
        );

        AccountMemoryStore::forget_account(&store, &alice)
            .await
            .unwrap();
        assert!(
            store
                .search(&alice, "async rust kubernetes cat", 10)
                .await
                .unwrap()
                .is_empty(),
            "forget_account deletes the account's entries"
        );
        assert_eq!(
            store.search(&bob, "async rust", 10).await.unwrap().len(),
            1,
            "forget_account leaves other accounts alone"
        );
    }

    /// Cache-specific rules for `Cached<_, C>` over a session store:
    /// read-after-write, survival of a cache outage, no stale reads after
    /// deletion. `cache` must be the same (shared) cache the store wraps, and
    /// `set_down` must toggle its availability.
    pub async fn cached_sessions<S: SessionStore>(store: S, set_down: impl Fn(bool)) {
        let (alice, _, s1, _) = ids();
        store.append(&alice, &s1, Turn::user("one")).await.unwrap();
        assert_eq!(store.recent(&alice, &s1, 5).await.unwrap().len(), 1);
        store.append(&alice, &s1, Turn::user("two")).await.unwrap();
        assert_eq!(
            store.recent(&alice, &s1, 5).await.unwrap().len(),
            2,
            "a read after a write sees the write, even when the first read was cached"
        );

        set_down(true);
        store
            .append(&alice, &s1, Turn::user("three"))
            .await
            .unwrap();
        assert_eq!(
            store.recent(&alice, &s1, 5).await.unwrap().len(),
            3,
            "reads and writes keep working while the cache is down"
        );
        set_down(false);

        store.recent(&alice, &s1, 5).await.unwrap();
        SessionStore::forget_account(&store, &alice).await.unwrap();
        assert!(
            store.recent(&alice, &s1, 5).await.unwrap().is_empty(),
            "no cached copy survives forget_account"
        );

        store
            .append(&alice, &s1, Turn::user("again"))
            .await
            .unwrap();
        store.recent(&alice, &s1, 5).await.unwrap();
        set_down(true);
        assert!(
            SessionStore::forget_account(&store, &alice).await.is_err(),
            "forget_account fails rather than claim success while cached copies may remain"
        );
        set_down(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_codec_round_trips_awkward_text() {
        let turns = vec![
            Turn::user("a:b\\c\nd"),
            Turn::assistant(""),
            Turn::assistant("\\n literal"),
        ];
        assert_eq!(decode_turns(&encode_turns(&turns)).unwrap(), turns);
        assert_eq!(decode_turns("").unwrap(), Vec::new());
        assert!(decode_turns("x:bad role").is_none());
    }

    #[test]
    fn cache_keys_cannot_collide_across_ids() {
        let a = session_key(&AccountId::new("a:b"), &SessionId::new("c"));
        let b = session_key(&AccountId::new("a"), &SessionId::new("b:c"));
        assert_ne!(a, b);
        assert!(!b.starts_with(&account_prefix(&AccountId::new("a:b"))));
        assert!(
            !session_key(&AccountId::new("ab"), &SessionId::new("x"))
                .starts_with(&account_prefix(&AccountId::new("a")))
        );
    }
}
