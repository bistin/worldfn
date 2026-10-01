//! Stopping long-running work: [`CancelToken`] and deadlines.
//!
//! Implemented with `std` only, like [`emit`](crate::emit), so it works under
//! any async runtime. A token is cheap to clone; cancelling any clone wakes
//! every task waiting on it.

use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Why work was stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// [`CancelToken::cancel`] was called.
    Cancelled,
    /// A time limit passed.
    DeadlineExceeded,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StopReason::Cancelled => "cancelled",
            StopReason::DeadlineExceeded => "deadline exceeded",
        })
    }
}

#[derive(Default)]
struct State {
    stopped: Option<StopReason>,
    wakers: Vec<Waker>,
}

/// A shareable stop signal. Pass a clone to the work and keep one to cancel
/// it, e.g. when a client disconnects or the user presses Ctrl-C.
///
/// Stopping only abandons futures at their next await point. Work that
/// lives outside the process, such as a command on a remote machine, needs
/// its own cleanup.
#[derive(Clone, Default)]
pub struct CancelToken {
    state: Arc<Mutex<State>>,
}

impl std::fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelToken")
            .field("stopped", &self.reason())
            .finish()
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.stop(StopReason::Cancelled);
    }

    /// The first reason wins; later calls change nothing.
    pub(crate) fn stop(&self, reason: StopReason) {
        let wakers = {
            let mut state = self.lock();
            if state.stopped.is_some() {
                return;
            }
            state.stopped = Some(reason);
            std::mem::take(&mut state.wakers)
        };
        for waker in wakers {
            waker.wake();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.reason().is_some()
    }

    pub fn reason(&self) -> Option<StopReason> {
        self.lock().stopped
    }

    fn poll_stopped(&self, cx: &mut Context<'_>) -> Poll<StopReason> {
        let mut state = self.lock();
        if let Some(reason) = state.stopped {
            return Poll::Ready(reason);
        }
        if !state.wakers.iter().any(|w| w.will_wake(cx.waker())) {
            state.wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run `future` until it finishes or any of `signals` stops.
    pub async fn race<F: Future>(
        signals: &[&CancelToken],
        future: F,
    ) -> Result<F::Output, StopReason> {
        let mut future = pin!(future);
        std::future::poll_fn(|cx| {
            for signal in signals {
                if let Poll::Ready(reason) = signal.poll_stopped(cx) {
                    return Poll::Ready(Err(reason));
                }
            }
            future.as_mut().poll(cx).map(Ok)
        })
        .await
    }
}

/// Stops `token` with [`StopReason::DeadlineExceeded`] after `timeout`.
/// Dropping the guard before then ends the timer thread.
#[cfg_attr(not(feature = "structured"), allow(dead_code))]
pub(crate) struct Deadline {
    at: Instant,
    done: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

#[cfg_attr(not(feature = "structured"), allow(dead_code))]
impl Deadline {
    pub(crate) fn start(timeout: Duration, token: CancelToken) -> Self {
        let at = Instant::now() + timeout;
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        let thread = std::thread::spawn(move || {
            loop {
                if flag.load(Ordering::Acquire) {
                    return;
                }
                let now = Instant::now();
                if now >= at {
                    token.stop(StopReason::DeadlineExceeded);
                    return;
                }
                std::thread::park_timeout(at - now);
            }
        });
        Self {
            at,
            done,
            thread: Some(thread),
        }
    }

    pub(crate) fn passed(&self) -> bool {
        Instant::now() >= self.at
    }
}

impl Drop for Deadline {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
        }
    }
}
