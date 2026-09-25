#pragma once
// dye/fence.hpp — "the GPU has finished", as a file descriptor.
//
// Why this exists
// ---------------
// Two programs share a buffer: a client draws into it, a compositor reads
// it. Something has to say when the drawing is done.
//
// IMPLICIT sync — the default on EGL — has the kernel guess from which
// buffers were touched. It works, and it serialises far more than it needs
// to: a compositor that reads a buffer waits for every operation the client
// queued, not just the one that filled it. Under load that is most of the
// jank people describe as "the compositor feels slow".
//
// EXPLICIT sync passes the answer around as a value. The client hands over
// an *acquire* fence ("wait for this before reading"); the compositor hands
// back a *release* fence ("I am done, you may reuse the buffer"). Nobody
// blocks: the waiting happens on the GPU, in the order the work was
// submitted.
//
// This is what linux-drm-syncobj-v1 is, and it is why dye is on Vulkan.
// Measured here, every combination a compositor needs is available:
//
//     binary   + SYNC_FD      export=yes  import=yes
//     binary   + OPAQUE_FD    export=yes  import=yes
//     timeline + OPAQUE_FD    export=yes  import=yes
//
// The type
// --------
// A Fence is an owned descriptor and nothing else. It is move-only, because
// a fence with two owners is a descriptor closed twice; it is not copyable
// for the same reason. An empty Fence is meaningful — "nothing to wait for"
// — and is what a client that does not use explicit sync gives you.

#include <cstdint>
#include <utility>

#include <jaal/core/error.hpp>
#include <jaal/platform/handle.hpp>

namespace dye {

/// A point in time on the GPU.
///
/// Wraps a sync_file descriptor: the kernel's own representation, which is
/// what makes it shareable with a client, with KMS, and across processes.
class Fence {
public:
    Fence() = default;

    /// Adopt a descriptor. Explicit, so the places where a fence is claimed
    /// out of thin air are greppable.
    explicit Fence(jaal::platform::owned_handle fd) : fd_(std::move(fd)) {}

    /// Is there anything to wait for?
    ///
    /// An empty fence is NOT an error. A client using implicit sync sends no
    /// fence at all, and the right response is to carry on, not to fail.
    [[nodiscard]] bool valid() const noexcept { return fd_.valid(); }
    explicit operator bool() const noexcept { return valid(); }

    /// Lend the descriptor: for handing to KMS or to Vulkan, neither of
    /// which takes ownership.
    [[nodiscard]] jaal::platform::borrowed_handle borrow() const noexcept {
        return fd_.borrow();
    }

    /// Give the descriptor away. [[nodiscard]] because dropping the result
    /// leaks it.
    [[nodiscard]] jaal::platform::owned_handle take() noexcept { return std::move(fd_); }

    /// Block until the GPU reaches this point.
    ///
    /// The escape hatch, not the normal path. A compositor should hand the
    /// fence to whatever waits next — the GPU or the display — rather than
    /// blocking its own loop on it. Here for teardown, for tests, and for
    /// the read-back path that is already slow.
    ///
    /// `timeout_ms` of -1 waits forever; anything else returns
    /// errc::timed_out, because a wedged GPU must not hang a compositor with
    /// no way out.
    [[nodiscard]] jaal::result<void> wait(int timeout_ms = -1) const;

    /// Has it already been signalled? Never blocks.
    [[nodiscard]] bool ready() const;

    /// Both, as one fence: signalled when BOTH are.
    ///
    /// A frame that reads three client buffers has three acquire fences and
    /// one thing to wait for. The kernel merges sync_files natively, so this
    /// costs nothing and saves the renderer keeping a list.
    [[nodiscard]] static jaal::result<Fence> merge(const Fence& a, const Fence& b);

private:
    jaal::platform::owned_handle fd_{};
};

}  // namespace dye
