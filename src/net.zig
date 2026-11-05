//! Network loading and inference wrapper.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;
const Mat = @import("mat.zig").Mat;
const Option = @import("option.zig").Option;
const layer = @import("layer.zig");
const DataReader = @import("datareader.zig").DataReader;

/// Owns an `ncnn::Net`.
pub const Net = struct {
    handle: c.ncnn_net_t,

    /// Creates an empty net.
    pub fn init() Error!Net {
        return .{ .handle = c.ncnn_net_create() orelse return Error.CreateNetFailed };
    }

    /// Releases the net.
    pub fn deinit(self: *Net) void {
        if (self.handle) |handle| c.ncnn_net_destroy(handle);
        self.handle = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: Net) c.ncnn_net_t {
        return self.handle;
    }

    /// Copies the option into the net; caller keeps ownership.
    pub fn setOption(self: Net, option: Option) void {
        c.ncnn_net_set_option(self.handle, option.handle);
    }

    /// Sets the thread count without allocating an option.
    pub fn setNumThreads(self: Net, num_threads: i32) void {
        c.ncnn_option_set_num_threads(c.ncnn_net_get_option(self.handle), num_threads);
    }

    /// Loads the layer graph from a `.param` file.
    pub fn loadParamFile(self: Net, path: []const u8) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_z = try toPathZ(&buf, path);
        if (c.ncnn_net_load_param(self.handle, path_z.ptr) != 0) return Error.LoadParamFailed;
    }

    /// Loads the layer graph from a `.param` string.
    pub fn loadParamMemory(self: Net, param: [:0]const u8) Error!void {
        if (c.ncnn_net_load_param_memory(self.handle, param.ptr) != 0) return Error.LoadParamFailed;
    }

    /// Loads weights from a `.bin` file.
    pub fn loadModelFile(self: Net, path: []const u8) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_z = try toPathZ(&buf, path);
        if (c.ncnn_net_load_model(self.handle, path_z.ptr) != 0) return Error.LoadModelFailed;
    }

    /// Loads weights from memory and returns the bytes consumed.
    ///
    /// The consumed count matters when one buffer holds several models.
    pub fn loadModelMemory(self: Net, data: []const u8) Error!usize {
        const consumed = c.ncnn_net_load_model_memory(self.handle, data.ptr);
        if (consumed == 0) return Error.LoadModelFailed;
        return consumed;
    }

    /// Loads a binary `.param` file.
    pub fn loadParamBinFile(self: Net, path: []const u8) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_z = try toPathZ(&buf, path);
        if (c.ncnn_net_load_param_bin(self.handle, path_z.ptr) != 0) return Error.LoadParamFailed;
    }

    /// Loads the text `.param` graph from a data reader.
    pub fn loadParamDatareader(self: Net, reader: DataReader) Error!void {
        if (c.ncnn_net_load_param_datareader(self.handle, reader.handle) != 0) return Error.LoadParamFailed;
    }

    /// Loads a binary `.param` graph from a data reader.
    pub fn loadParamBinDatareader(self: Net, reader: DataReader) Error!void {
        if (c.ncnn_net_load_param_bin_datareader(self.handle, reader.handle) != 0) return Error.LoadParamFailed;
    }

    /// Loads weights from a data reader.
    pub fn loadModelDatareader(self: Net, reader: DataReader) Error!void {
        if (c.ncnn_net_load_model_datareader(self.handle, reader.handle) != 0) return Error.LoadModelFailed;
    }

    /// Blob index of the i-th network input.
    pub fn inputIndex(self: Net, index: usize) i32 {
        return c.ncnn_net_get_input_index(self.handle, @intCast(index));
    }

    /// Blob index of the i-th network output.
    pub fn outputIndex(self: Net, index: usize) i32 {
        return c.ncnn_net_get_output_index(self.handle, @intCast(index));
    }

    /// Releases every loaded layer and blob.
    pub fn clear(self: Net) void {
        c.ncnn_net_clear(self.handle);
    }

    /// Number of network inputs.
    pub fn inputCount(self: Net) usize {
        return @intCast(c.ncnn_net_get_input_count(self.handle));
    }

    /// Number of network outputs.
    pub fn outputCount(self: Net) usize {
        return @intCast(c.ncnn_net_get_output_count(self.handle));
    }

    /// Input blob name at `index`.
    pub fn inputName(self: Net, index: usize) ?[:0]const u8 {
        if (index >= self.inputCount()) return null;
        return cString(c.ncnn_net_get_input_name(self.handle, @intCast(index)));
    }

    /// Output blob name at `index`.
    pub fn outputName(self: Net, index: usize) ?[:0]const u8 {
        if (index >= self.outputCount()) return null;
        return cString(c.ncnn_net_get_output_name(self.handle, @intCast(index)));
    }

    /// Registers a custom layer factory by type name.
    pub fn registerCustomLayerByType(
        self: Net,
        layer_type: []const u8,
        creator: layer.Creator,
        destroyer: layer.Destroyer,
        userdata: ?*anyopaque,
    ) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const type_z = try toPathZ(&buf, layer_type);
        c.ncnn_net_register_custom_layer_by_type(self.handle, type_z.ptr, creator, destroyer, userdata);
    }

    /// Registers a custom layer factory by type index.
    pub fn registerCustomLayerByTypeIndex(
        self: Net,
        typeindex: i32,
        creator: layer.Creator,
        destroyer: layer.Destroyer,
        userdata: ?*anyopaque,
    ) void {
        c.ncnn_net_register_custom_layer_by_typeindex(self.handle, typeindex, creator, destroyer, userdata);
    }

    /// Creates an extractor; the net must outlive it.
    pub fn createExtractor(self: Net) Error!Extractor {
        return .{ .handle = c.ncnn_extractor_create(self.handle) orelse return Error.CreateExtractorFailed };
    }
};

/// Owns an `ncnn::Extractor`.
///
/// One extractor runs one inference pass over a fixed set of inputs.
pub const Extractor = struct {
    handle: c.ncnn_extractor_t,

    /// Releases the extractor.
    pub fn deinit(self: *Extractor) void {
        if (self.handle) |handle| c.ncnn_extractor_destroy(handle);
        self.handle = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: Extractor) c.ncnn_extractor_t {
        return self.handle;
    }

    /// Copies the option into the extractor.
    pub fn setOption(self: Extractor, option: Option) void {
        c.ncnn_extractor_set_option(self.handle, option.handle);
    }

    /// Feeds an input blob by name.
    pub fn input(self: Extractor, name: []const u8, mat: Mat) Error!void {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const name_z = try toPathZ(&buf, name);
        if (c.ncnn_extractor_input(self.handle, name_z.ptr, mat.handle) != 0) return Error.InputFailed;
    }

    /// Feeds an input blob by index.
    pub fn inputIndex(self: Extractor, index: i32, mat: Mat) Error!void {
        if (c.ncnn_extractor_input_index(self.handle, index, mat.handle) != 0) return Error.InputFailed;
    }

    /// Runs the graph and returns an output blob by name.
    pub fn extract(self: Extractor, name: []const u8) Error!Mat {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const name_z = try toPathZ(&buf, name);

        var mat: c.ncnn_mat_t = null;
        const ret = c.ncnn_extractor_extract(self.handle, name_z.ptr, &mat);
        return takeResult(ret, mat);
    }

    /// Runs the graph and returns an output blob by index.
    pub fn extractIndex(self: Extractor, index: i32) Error!Mat {
        var mat: c.ncnn_mat_t = null;
        const ret = c.ncnn_extractor_extract_index(self.handle, index, &mat);
        return takeResult(ret, mat);
    }
};

/// The C API allocates a Mat even on failure; free it in that case.
fn takeResult(ret: c_int, handle: c.ncnn_mat_t) Error!Mat {
    if (ret != 0) {
        if (handle) |mat| c.ncnn_mat_destroy(mat);
        return Error.ExtractFailed;
    }
    return .{ .handle = handle orelse return Error.ExtractFailed };
}

/// NUL-terminates into a stack buffer; ncnn wants `const char*`.
fn toPathZ(buf: []u8, path: []const u8) Error![:0]const u8 {
    if (path.len >= buf.len) return Error.PathTooLong;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    return buf[0..path.len :0];
}

/// Converts a nullable C string, the caller must not free it.
fn cString(ptr: [*c]const u8) ?[:0]const u8 {
    if (ptr == null) return null;
    return std.mem.sliceTo(ptr[0..], 0);
}
