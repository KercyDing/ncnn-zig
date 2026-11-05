//! Data reader wrapper for loading models from custom sources.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

pub const ScanFn = *const fn (
    handle: c.ncnn_datareader_t,
    format: [*c]const u8,
    p: ?*anyopaque,
) callconv(.c) c_int;
pub const ReadFn = *const fn (
    handle: c.ncnn_datareader_t,
    buf: ?*anyopaque,
    size: usize,
) callconv(.c) usize;

/// Owns an `ncnn_datareader_t`.
pub const DataReader = struct {
    handle: c.ncnn_datareader_t,
    /// ncnn stores a reference to the cursor, so it must be heap-stable.
    memory_cursor: ?*[*c]const u8 = null,

    /// Creates a reader with no behaviour; install `setRead` and `setScan`.
    pub fn init() Error!DataReader {
        return .{ .handle = c.ncnn_datareader_create() orelse return Error.CreateDataReaderFailed };
    }

    /// Reads from `data`, which must outlive the reader.
    pub fn fromMemory(data: []const u8) Error!DataReader {
        // The reader keeps a reference, not a copy, of this cursor.
        const cursor = std.heap.c_allocator.create([*c]const u8) catch return Error.OutOfMemory;
        cursor.* = data.ptr;

        const handle = c.ncnn_datareader_create_from_memory(cursor) orelse {
            std.heap.c_allocator.destroy(cursor);
            return Error.CreateDataReaderFailed;
        };
        return .{ .handle = handle, .memory_cursor = cursor };
    }

    /// Releases the reader.
    pub fn deinit(self: *DataReader) void {
        if (self.handle) |handle| c.ncnn_datareader_destroy(handle);
        if (self.memory_cursor) |cursor| std.heap.c_allocator.destroy(cursor);
        self.handle = null;
        self.memory_cursor = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: DataReader) c.ncnn_datareader_t {
        return self.handle;
    }

    /// Installs the byte-read callback.
    pub fn setRead(self: DataReader, callback: ReadFn) void {
        self.cStruct().read = callback;
    }

    /// Installs the text-scan callback used by `.param` parsing.
    pub fn setScan(self: DataReader, callback: ScanFn) void {
        self.cStruct().scan = callback;
    }

    /// Raw struct access for fields the wrapper does not name yet.
    pub fn cStruct(self: DataReader) *c.struct___ncnn_datareader_t {
        return self.handle orelse unreachable;
    }
};
