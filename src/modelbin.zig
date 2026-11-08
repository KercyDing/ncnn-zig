//! Model weight reader wrapper around `ncnn::ModelBin`.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;
const Mat = @import("mat.zig").Mat;
const DataReader = @import("datareader.zig").DataReader;

/// Owns an `ncnn_modelbin_t`.
pub const ModelBin = struct {
    handle: c.ncnn_modelbin_t,

    /// Reads weights through `reader`, which must outlive the modelbin.
    pub fn fromDataReader(reader: DataReader) Error!ModelBin {
        const handle = c.ncnn_modelbin_create_from_datareader(reader.handle);
        return .{ .handle = handle orelse return Error.CreateModelBinFailed };
    }

    /// Releases the modelbin.
    pub fn deinit(self: *ModelBin) void {
        if (self.handle) |handle| c.ncnn_modelbin_destroy(handle);
        self.handle = null;
    }

    /// Transfers ownership and clears the source wrapper.
    pub fn move(self: *ModelBin) ModelBin {
        const result = self.*;
        self.handle = null;
        return result;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: ModelBin) c.ncnn_modelbin_t {
        return self.handle;
    }

    /// Loads a 1D weight tensor; `type_id` uses ncnn's ModelBin type codes.
    pub fn load1d(self: ModelBin, w: usize, type_id: i32) Error!Mat {
        const load = self.cStruct().load_1d orelse return Error.CreateMatFailed;
        const handle = load(self.handle, try dim(w), type_id);
        return .{ .handle = handle orelse return Error.CreateMatFailed };
    }

    /// Loads a 2D weight tensor; `type_id` uses ncnn's ModelBin type codes.
    pub fn load2d(self: ModelBin, w: usize, h: usize, type_id: i32) Error!Mat {
        const load = self.cStruct().load_2d orelse return Error.CreateMatFailed;
        const handle = load(self.handle, try dim(w), try dim(h), type_id);
        return .{ .handle = handle orelse return Error.CreateMatFailed };
    }

    /// Loads a 3D weight tensor; `type_id` uses ncnn's ModelBin type codes.
    pub fn load3d(self: ModelBin, w: usize, h: usize, channels: usize, type_id: i32) Error!Mat {
        const load = self.cStruct().load_3d orelse return Error.CreateMatFailed;
        const handle = load(self.handle, try dim(w), try dim(h), try dim(channels), type_id);
        return .{ .handle = handle orelse return Error.CreateMatFailed };
    }

    /// Raw struct access for fields the wrapper does not name yet.
    pub fn cStruct(self: ModelBin) *c.struct___ncnn_modelbin_t {
        return self.handle orelse unreachable;
    }
};

/// ncnn sizes are `int`.
fn dim(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse Error.OutOfMemory;
}
