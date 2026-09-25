// fence.cpp — sync_file, through the kernel's own ioctls.
//
// A sync_file is a descriptor you can poll(). That is the whole interface,
// and it is why explicit sync composes with an event loop: a fence is just
// another thing jaal can watch.
#include "dye/fence.hpp"

#include <linux/sync_file.h>
#include <poll.h>
#include <sys/ioctl.h>

#include <cerrno>
#include <cstring>

namespace dye {

namespace {

auto fail(std::errc code, std::string_view what) {
    return std::unexpected(jaal::error::make(code, what));
}
auto fail_errno(std::string_view what) {
    return std::unexpected(jaal::error::from_errno(errno, what));
}

}  // namespace

jaal::result<void> Fence::wait(int timeout_ms) const {
    if (!fd_.valid()) return {};   // nothing to wait for is not a failure

    pollfd p{fd_.get(), POLLIN, 0};
    int r;
    do {
        r = ::poll(&p, 1, timeout_ms);
    } while (r < 0 && errno == EINTR);

    if (r < 0) return fail_errno("waiting on a fence failed");
    if (r == 0) return fail(std::errc::timed_out, "the GPU did not signal in time");
    return {};
}

bool Fence::ready() const {
    if (!fd_.valid()) return true;   // nothing to wait for is already done
    pollfd p{fd_.get(), POLLIN, 0};
    return ::poll(&p, 1, 0) == 1;
}

jaal::result<Fence> Fence::merge(const Fence& a, const Fence& b) {
    // Either being empty makes the answer the other one: a frame with one
    // real acquire fence and two clients using implicit sync waits for
    // exactly the one thing there is to wait for.
    if (!a.valid()) {
        if (!b.valid()) return Fence{};
        auto dup = b.fd_.duplicate();
        if (!dup) return std::unexpected(dup.error());
        return Fence{std::move(*dup)};
    }
    if (!b.valid()) {
        auto dup = a.fd_.duplicate();
        if (!dup) return std::unexpected(dup.error());
        return Fence{std::move(*dup)};
    }

    sync_merge_data data{};
    std::strncpy(data.name, "dye", sizeof data.name - 1);
    data.fd2 = b.fd_.get();
    data.fence = -1;
    if (::ioctl(a.fd_.get(), SYNC_IOC_MERGE, &data) != 0)
        return fail_errno("merging two fences failed");

    return Fence{jaal::platform::owned_handle{data.fence}};
}

}  // namespace dye
