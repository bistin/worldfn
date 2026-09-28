//! Streaming events out of a running agent, independent of any web framework.
//!
//! The caller creates a [`channel`], puts the [`Emitter`] into the invocation
//! [`Scope`], and gives the [`EventStream`] to whatever delivers events: an
//! SSE response in a web framework adapter, a terminal printer, or a `Vec`
//! in a test. The agent declares the capability with an [`Emit<E>`] parameter.

use std::any::type_name;
use std::collections::VecDeque;
use std::fmt;
use std::future::{Future, Ready, ready};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context as TaskContext, Poll, Waker};

use crate::{AgentParam, AgentWorld, ParamError, Requirement, Scope};

struct Queue<E> {
    events: VecDeque<E>,
    waker: Option<Waker>,
    receiver_alive: bool,
}

struct Shared<E> {
    queue: Mutex<Queue<E>>,
    senders: AtomicUsize,
}

impl<E> Shared<E> {
    fn lock(&self) -> MutexGuard<'_, Queue<E>> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Create an unbounded event channel.
///
/// The stream ends once every [`Emitter`] clone is dropped, which happens
/// when the agent run holding it finishes. There is no backpressure: events
/// queue in memory until read.
pub fn channel<E>() -> (Emitter<E>, EventStream<E>) {
    let shared = Arc::new(Shared {
        queue: Mutex::new(Queue {
            events: VecDeque::new(),
            waker: None,
            receiver_alive: true,
        }),
        senders: AtomicUsize::new(1),
    });
    (
        Emitter {
            shared: shared.clone(),
        },
        EventStream { shared },
    )
}

/// The sending half. Put it in the [`Scope`] with `Scope::with(emitter)`.
pub struct Emitter<E> {
    shared: Arc<Shared<E>>,
}

impl<E> Emitter<E> {
    /// Queue an event. Returns `false` if the receiving side is gone (e.g. the
    /// client disconnected), so a long-running agent can stop early.
    pub fn send(&self, event: E) -> bool {
        let mut queue = self.shared.lock();
        if !queue.receiver_alive {
            return false;
        }
        queue.events.push_back(event);
        if let Some(waker) = queue.waker.take() {
            waker.wake();
        }
        true
    }

    /// Whether anyone is still listening.
    pub fn is_open(&self) -> bool {
        self.shared.lock().receiver_alive
    }
}

impl<E> Clone for Emitter<E> {
    fn clone(&self) -> Self {
        self.shared.senders.fetch_add(1, Ordering::AcqRel);
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl<E> Drop for Emitter<E> {
    fn drop(&mut self) {
        if self.shared.senders.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Last sender: wake the reader so it observes the end.
            if let Some(waker) = self.shared.lock().waker.take() {
                waker.wake();
            }
        }
    }
}

impl<E> fmt::Debug for Emitter<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Emitter")
            .field("open", &self.is_open())
            .finish()
    }
}

/// The receiving half: a runtime-agnostic async sequence of events.
///
/// Framework adapters turn it into their stream type; anything else can
/// loop on [`recv`](Self::recv).
pub struct EventStream<E> {
    shared: Arc<Shared<E>>,
}

impl<E> EventStream<E> {
    /// The next event, or `None` once all emitters are dropped and the queue
    /// is drained.
    pub fn recv(&mut self) -> Recv<'_, E> {
        Recv { stream: self }
    }

    /// Poll for the next event; the building block for `Stream` impls.
    pub fn poll_next(&mut self, cx: &mut TaskContext<'_>) -> Poll<Option<E>> {
        let mut queue = self.shared.lock();
        if let Some(event) = queue.events.pop_front() {
            return Poll::Ready(Some(event));
        }
        if self.shared.senders.load(Ordering::Acquire) == 0 {
            return Poll::Ready(None);
        }
        queue.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

impl<E> Drop for EventStream<E> {
    fn drop(&mut self) {
        let mut queue = self.shared.lock();
        queue.receiver_alive = false;
        queue.events.clear();
    }
}

/// Future returned by [`EventStream::recv`].
pub struct Recv<'a, E> {
    stream: &'a mut EventStream<E>,
}

impl<E> Future for Recv<'_, E> {
    type Output = Option<E>;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<E>> {
        self.get_mut().stream.poll_next(cx)
    }
}

/// Parameter: permission to stream events of type `E` to whoever invoked the
/// agent. The [`Emitter`] comes from the invocation [`Scope`]; a missing one
/// is a caller error, reported before the body runs.
pub struct Emit<E>(Emitter<E>);

impl<E> Emit<E> {
    pub fn send(&self, event: E) -> bool {
        self.0.send(event)
    }

    pub fn is_open(&self) -> bool {
        self.0.is_open()
    }
}

impl<E> Clone for Emit<E> {
    fn clone(&self) -> Self {
        Emit(self.0.clone())
    }
}

impl<E: Send + 'static> AgentParam for Emit<E> {
    type State = ();
    type Future = Ready<Result<Self, ParamError>>;

    fn describe(out: &mut Vec<Requirement>) {
        out.push(Requirement::Emit {
            type_name: type_name::<E>(),
        });
    }

    fn init(_world: &AgentWorld) -> Result<(), Vec<Requirement>> {
        Ok(())
    }

    fn resolve(_state: &mut (), _world: &AgentWorld, scope: &Scope) -> Self::Future {
        ready(
            scope
                .get::<Emitter<E>>()
                .map(|emitter| Emit((*emitter).clone()))
                .ok_or_else(|| ParamError::missing_from_scope(type_name::<Self>())),
        )
    }
}

/// One server-sent event, in a framework-neutral form.
///
/// Implement [`SseEvent`] for your event type to say how it goes on the
/// wire; adapters and [`SseFrame::encode`] do the rest.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SseFrame {
    /// The `event:` field; `None` means the default `message` event.
    pub event: Option<String>,
    /// The `data:` payload. Newlines become multiple `data:` lines.
    pub data: String,
    pub id: Option<String>,
}

impl SseFrame {
    pub fn new(event: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            event: Some(event.into()),
            data: data.into(),
            id: None,
        }
    }

    /// The wire format, ending with the blank line that terminates a frame.
    pub fn encode(&self) -> String {
        let mut out = String::new();
        if let Some(id) = &self.id {
            out.push_str(&format!("id: {}\n", single_line(id)));
        }
        if let Some(event) = &self.event {
            out.push_str(&format!("event: {}\n", single_line(event)));
        }
        for line in self.data.split('\n') {
            out.push_str("data: ");
            out.push_str(line.strip_suffix('\r').unwrap_or(line));
            out.push('\n');
        }
        out.push('\n');
        out
    }
}

fn single_line(s: &str) -> String {
    s.replace(['\r', '\n'], " ")
}

/// How an event type is represented as a server-sent event.
pub trait SseEvent {
    fn to_sse(&self) -> SseFrame;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn events_flow_until_the_last_emitter_drops() {
        let (tx, mut rx) = channel();
        let tx2 = tx.clone();
        let task = tokio::spawn(async move {
            assert!(tx.send(1));
            tokio::task::yield_now().await;
            assert!(tx2.send(2));
        });
        assert_eq!(rx.recv().await, Some(1));
        assert_eq!(rx.recv().await, Some(2));
        task.await.unwrap();
        assert_eq!(rx.recv().await, None);
    }

    #[test]
    fn send_reports_a_dropped_receiver() {
        let (tx, rx) = channel();
        assert!(tx.is_open());
        drop(rx);
        assert!(!tx.send("late"));
        assert!(!tx.is_open());
    }

    #[test]
    fn sse_encoding() {
        let frame = SseFrame {
            id: Some("7".into()),
            ..SseFrame::new("token", "line one\nline two")
        };
        assert_eq!(
            frame.encode(),
            "id: 7\nevent: token\ndata: line one\ndata: line two\n\n"
        );
        assert_eq!(
            SseFrame {
                data: "x".into(),
                ..Default::default()
            }
            .encode(),
            "data: x\n\n"
        );
    }
}
