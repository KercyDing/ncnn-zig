//! Allocator wrapper for blob and workspace memory.

const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// Owns an `ncnn_allocator_t`.
pub const Allocator = struct {
    handle: c.ncnn_allocator_t,

    /// Creates a pool allocator, safe for multi-threaded use.
    pub fn initPool() Error!Allocator {
        return .{ .handle = c.ncnn_allocator_create_pool_allocator() orelse return Error.CreateAllocatorFailed };
    }

    /// Creates a pool allocator without locking, for single-threaded use.
    pub fn initUnlockedPool() Error!Allocator {
        return .{ .handle = c.ncnn_allocator_create_unlocked_pool_allocator() orelse return Error.CreateAllocatorFailed };
    }

    /// Releases the allocator and its pooled blocks.
    pub fn deinit(self: *Allocator) void {
        if (self.handle) |handle| c.ncnn_allocator_destroy(handle);
        self.handle = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: Allocator) c.ncnn_allocator_t {
        return self.handle;
    }

    /// Allocates `size` bytes through the pool.
    pub fn alloc(self: Allocator, size: usize) ?*anyopaque {
        const fast_malloc = self.cStruct().fast_malloc orelse return null;
        return fast_malloc(self.handle, size);
    }

    /// Returns a block obtained from `alloc`.
    pub fn free(self: Allocator, ptr: ?*anyopaque) void {
        const fast_free = self.cStruct().fast_free orelse return;
        fast_free(self.handle, ptr);
    }

    /// Raw struct access for fields the wrapper does not name yet.
    pub fn cStruct(self: Allocator) *c.struct___ncnn_allocator_t {
        return self.handle orelse unreachable;
    }
};
