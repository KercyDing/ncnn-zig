//! Parameter dictionary wrapper around `ncnn::ParamDict`.

const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;
const Mat = @import("mat.zig").Mat;

/// Owns an `ncnn::ParamDict`, the id-keyed bag used by layer parameters.
pub const ParamDict = struct {
    handle: c.ncnn_paramdict_t,

    /// Creates an empty dictionary.
    pub fn init() Error!ParamDict {
        return .{ .handle = c.ncnn_paramdict_create() orelse return Error.CreateParamDictFailed };
    }

    /// Releases the dictionary.
    pub fn deinit(self: *ParamDict) void {
        if (self.handle) |handle| c.ncnn_paramdict_destroy(handle);
        self.handle = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: ParamDict) c.ncnn_paramdict_t {
        return self.handle;
    }

    /// Stores an integer.
    pub fn setInt(self: ParamDict, id: i32, value: i32) void {
        c.ncnn_paramdict_set_int(self.handle, id, value);
    }

    /// Stores a float.
    pub fn setFloat(self: ParamDict, id: i32, value: f32) void {
        c.ncnn_paramdict_set_float(self.handle, id, value);
    }

    /// Stores an array value, copied into the dictionary.
    pub fn setArray(self: ParamDict, id: i32, value: Mat) void {
        c.ncnn_paramdict_set_array(self.handle, id, value.handle);
    }

    /// Returns the integer, or `default` for a missing id.
    pub fn getInt(self: ParamDict, id: i32, default: i32) i32 {
        return c.ncnn_paramdict_get_int(self.handle, id, default);
    }

    /// Returns the float, or `default` for a missing id.
    pub fn getFloat(self: ParamDict, id: i32, default: f32) f32 {
        return c.ncnn_paramdict_get_float(self.handle, id, default);
    }

    /// Returns a copy of the array value; caller owns it.
    pub fn getArray(self: ParamDict, id: i32) Error!Mat {
        // The C API needs a concrete default mat, never null.
        const empty = c.ncnn_mat_create() orelse return Error.CreateMatFailed;
        defer c.ncnn_mat_destroy(empty);

        const handle = c.ncnn_paramdict_get_array(self.handle, id, empty);
        return .{ .handle = handle orelse return Error.CreateMatFailed };
    }

    /// Returns the stored ncnn type id, 0 for an empty slot.
    pub fn typeId(self: ParamDict, id: i32) i32 {
        return c.ncnn_paramdict_get_type(self.handle, id);
    }
};
