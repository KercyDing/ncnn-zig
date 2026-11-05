const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;
const Mat = @import("mat.zig").Mat;
const Option = @import("option.zig").Option;

/// Owns an `ncnn::Net`.
pub const Net = struct {
    handle: c.ncnn_net_t,

    pub fn init() Error!Net {
        return .{ .handle = c.ncnn_net_create() orelse return Error.CreateNetFailed };
    }

    pub fn deinit(self: *Net) void {
        if (self.handle) |handle| c.ncnn_net_destroy(handle);
        self.handle = null;
    }

    pub fn raw(self: Net) c.ncnn_net_t {
        return self.handle;
    }

    /// Copies the option into the net. `option` stays owned by the caller.
    pub fn setOption(self: Net, option: Option) void {
        c.ncnn_net_set_option(self.handle, option.handle);
    }

    /// Sets the thread count without allocating an `Option`.
    pub fn setNumThreads(self: Net, num_threads: i32) void {
        c.ncnn_option_set_num_threads(c.ncnn_net_get_option(self.handle), num_threads);
    }

    pub fn loadParamFile(self: Net, path: []const u8) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_z = try toPathZ(&buf, path);
        if (c.ncnn_net_load_param(self.handle, path_z.ptr) != 0) return Error.LoadParamFailed;
    }

    pub fn loadParamMemory(self: Net, param: [:0]const u8) Error!void {
        if (c.ncnn_net_load_param_memory(self.handle, param.ptr) != 0) return Error.LoadParamFailed;
    }

    pub fn loadModelFile(self: Net, path: []const u8) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_z = try toPathZ(&buf, path);
        if (c.ncnn_net_load_model(self.handle, path_z.ptr) != 0) return Error.LoadModelFailed;
    }

    /// Loads a weight blob from memory and returns the number of bytes consumed.
    pub fn loadModelMemory(self: Net, data: []const u8) Error!usize {
        const consumed = c.ncnn_net_load_model_memory(self.handle, data.ptr);
        if (consumed == 0) return Error.LoadModelFailed;
        return consumed;
    }

    pub fn clear(self: Net) void {
        c.ncnn_net_clear(self.handle);
    }

    pub fn inputCount(self: Net) usize {
        return @intCast(c.ncnn_net_get_input_count(self.handle));
    }

    pub fn outputCount(self: Net) usize {
        return @intCast(c.ncnn_net_get_output_count(self.handle));
    }

    pub fn inputName(self: Net, index: usize) ?[:0]const u8 {
        if (index >= self.inputCount()) return null;
        return cString(c.ncnn_net_get_input_name(self.handle, @intCast(index)));
    }

    pub fn outputName(self: Net, index: usize) ?[:0]const u8 {
        if (index >= self.outputCount()) return null;
        return cString(c.ncnn_net_get_output_name(self.handle, @intCast(index)));
    }

    /// Creates an extractor. The net must outlive it.
    pub fn createExtractor(self: Net) Error!Extractor {
        return .{ .handle = c.ncnn_extractor_create(self.handle) orelse return Error.CreateExtractorFailed };
    }
};

/// Owns an `ncnn::Extractor`.
pub const Extractor = struct {
    handle: c.ncnn_extractor_t,

    pub fn deinit(self: *Extractor) void {
        if (self.handle) |handle| c.ncnn_extractor_destroy(handle);
        self.handle = null;
    }

    pub fn raw(self: Extractor) c.ncnn_extractor_t {
        return self.handle;
    }

    pub fn setOption(self: Extractor, option: Option) void {
        c.ncnn_extractor_set_option(self.handle, option.handle);
    }

    pub fn input(self: Extractor, name: []const u8, mat: Mat) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const name_z = try toPathZ(&buf, name);
        if (c.ncnn_extractor_input(self.handle, name_z.ptr, mat.handle) != 0) return Error.InputFailed;
    }

    pub fn inputIndex(self: Extractor, index: i32, mat: Mat) Error!void {
        if (c.ncnn_extractor_input_index(self.handle, index, mat.handle) != 0) return Error.InputFailed;
    }

    pub fn extract(self: Extractor, name: []const u8) Error!Mat {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const name_z = try toPathZ(&buf, name);

        var mat: c.ncnn_mat_t = null;
        const ret = c.ncnn_extractor_extract(self.handle, name_z.ptr, &mat);
        return takeResult(ret, mat);
    }

    pub fn extractIndex(self: Extractor, index: i32) Error!Mat {
        var mat: c.ncnn_mat_t = null;
        const ret = c.ncnn_extractor_extract_index(self.handle, index, &mat);
        return takeResult(ret, mat);
    }
};

/// The C API hands out a fresh Mat even on failure; free it in that case.
fn takeResult(ret: c_int, handle: c.ncnn_mat_t) Error!Mat {
    if (ret != 0) {
        if (handle) |mat| c.ncnn_mat_destroy(mat);
        return Error.ExtractFailed;
    }
    return .{ .handle = handle orelse return Error.ExtractFailed };
}

fn toPathZ(buf: []u8, path: []const u8) Error![:0]const u8 {
    if (path.len >= buf.len) return Error.PathTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return buf[0..path.len :0];
}

fn cString(ptr: [*c]const u8) ?[:0]const u8 {
    if (ptr == null) return null;
    return std.mem.sliceTo(ptr[0..], 0);
}
