# Side effects, recovery and cancellation (VM work)

Rules that `worldfn-vm` and the tool loop follow before any VM tool is added.
They exist because VM tools change the world: a command that ran cannot be
"un-run", and a run that is repeated blindly repeats its effects.

## 1. One VM, one run

A VM (one guest) serves one agent run at a time. The adapter takes an
exclusive lease when a run starts and releases it after cleanup; a second run
on the same VM is refused, not queued. Everything a run creates (workspace
directory, processes, browser profile, screenshots) is owned by that run's id.

This keeps responsibility simple: whatever is on the guest during a run was
put there by that run, and cleanup knows exactly what to remove.

## 2. The loop never repeats a tool

`Toolbox::run_with` runs each tool call once. It does not retry tools and
does not retry model calls. When anything fails, the error carries every call
that already completed (`ToolLoopError::calls`), including a failed model call
(`ToolLoopError::Llm { error, calls }`). Those calls happened.

Calls that never ran are never executed later: a reply cut off at the length
limit or filtered (`Unfinished`), or with invalid ids (`InvalidTurn`), runs
none of its calls.

## 3. Recovery is the caller's decision

After an error the caller has the completed calls and, through them, what
changed. Resetting the VM undoes only what happened **inside** the VM. A run
can also change things outside it: a `git push`, an email, a payment, a call
to an external API, a file on a shared mount. A reset does not undo those,
and running again repeats them.

So the first question is where the effects went:

1. **All effects stayed in the resettable environment** (the guest's disk,
   processes and browser, with no outside writes): reset the VM (restore the
   snapshot or clone a fresh one) and run again from the task. This is the
   only case where re-running is safe without further checks.
2. **An effect reached the outside world, or a call's outcome is unknown**
   (the connection dropped mid-command): first check whether it happened,
   against the outside system itself (is the commit on the remote, was the
   email sent). Re-run only what is confirmed not to have happened. If it
   cannot be confirmed, stop and report; do not re-run.
3. **Continue the same conversation** only when every completed call is in
   the transcript the model will see, so it knows what already happened.
   Never re-send the original request as if nothing ran.

To keep the demo in case 1, the demo VM has no outbound network except what
setup explicitly allows (e.g. a local package cache), no host mounts and no
credentials for outside services. A task that needs outside effects (pushing
code, sending mail) is outside the first milestone; when one is added, its
tool must say so in its contract, and the runner treats any run that called
it as case 2.

A call whose outcome is unknown is reported as unknown. It is never assumed
to have failed, and never assumed not to have run.

## 4. Which actions may be retried

The adapter may retry only operations with no effect on the guest, and only
inside one tool call:

| Retry inside the adapter | Never retry automatically |
|---|---|
| reading a file, process status, screenshot, page observation | shell `run`/`spawn`, file writes, browser click/type, desktop click/key |
| connecting SSH (before anything was sent) | anything after the request reached the guest |

## 5. Cancellation and cleanup

Stopping the loop (timeout or `CancelToken`) only abandons the Rust future.
Guest work keeps running until the adapter stops it, so:

- every command runs in its own process group under the run's id;
- spawned processes get a time-to-live, enforced by a watchdog on the guest
  that also works if the host disappears;
- on stop, the adapter sends TERM to the run's process groups, waits briefly,
  sends KILL, closes the browser, then releases the lease;
- cleanup that cannot be confirmed (e.g. SSH is down) is reported as
  unconfirmed, and the VM must be reset before its next run.

Cleanup is an explicit async step the host calls on every exit path (success,
error, timeout, Ctrl-C). A `Drop` guard alone cannot do it, because cleanup
needs to talk to the guest.
