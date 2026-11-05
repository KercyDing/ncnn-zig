//! Custom layer wrapper around `ncnn::Layer`.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// Factory called by ncnn when the net instantiates the layer.
pub const Creator = *const fn (userdata: ?*anyopaque) callconv(.c) c.ncnn_layer_t;
/// Destroys a layer created by `Creator`.
pub const Destroyer = *const fn (handle: c.ncnn_layer_t, userdata: ?*anyopaque) callconv(.c) void;

pub const LoadParamFn = *const fn (handle: c.ncnn_layer_t, pd: c.ncnn_paramdict_t) callconv(.c) c_int;
pub const LoadModelFn = *const fn (handle: c.ncnn_layer_t, mb: c.ncnn_modelbin_t) callconv(.c) c_int;
pub const PipelineFn = *const fn (handle: c.ncnn_layer_t, opt: c.ncnn_option_t) callconv(.c) c_int;
pub const Forward1Fn = *const fn (
    handle: c.ncnn_layer_t,
    bottom_blob: c.ncnn_mat_t,
    top_blob: [*c]c.ncnn_mat_t,
    opt: c.ncnn_option_t,
) callconv(.c) c_int;
pub const ForwardNFn = *const fn (
    handle: c.ncnn_layer_t,
    bottom_blobs: [*c]const c.ncnn_mat_t,
    n: c_int,
    top_blobs: [*c]c.ncnn_mat_t,
    n2: c_int,
    opt: c.ncnn_option_t,
) callconv(.c) c_int;
pub const ForwardInplace1Fn = *const fn (
    handle: c.ncnn_layer_t,
    bottom_top_blob: c.ncnn_mat_t,
    opt: c.ncnn_option_t,
) callconv(.c) c_int;
pub const ForwardInplaceNFn = *const fn (
    handle: c.ncnn_layer_t,
    bottom_top_blobs: [*c]c.ncnn_mat_t,
    n: c_int,
    opt: c.ncnn_option_t,
) callconv(.c) c_int;

/// Owns an `ncnn_layer_t` created with `ncnn_layer_create`.
///
/// The C API reads behaviour flags from the struct, so set them before
/// returning the layer from a `Creator`.
pub const Layer = struct {
    handle: c.ncnn_layer_t,

    /// Creates a layer with no behaviour and default flags.
    pub fn init() Error!Layer {
        return .{ .handle = c.ncnn_layer_create() orelse return Error.CreateLayerFailed };
    }

    /// Creates a built-in layer by type name.
    pub fn createByType(layer_type: []const u8) Error!Layer {
        if (layer_type.len >= std.fs.max_path_bytes) return Error.PathTooLong;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        @memcpy(buf[0..layer_type.len], layer_type);
        buf[layer_type.len] = 0;

        const handle = c.ncnn_layer_create_by_type(buf[0..layer_type.len :0].ptr);
        return .{ .handle = handle orelse return Error.CreateLayerFailed };
    }

    /// Creates a built-in layer by type index.
    pub fn createByTypeIndex(typeindex: i32) Error!Layer {
        const handle = c.ncnn_layer_create_by_typeindex(typeindex);
        return .{ .handle = handle orelse return Error.CreateLayerFailed };
    }

    /// Releases the layer.
    pub fn deinit(self: *Layer) void {
        if (self.handle) |handle| c.ncnn_layer_destroy(handle);
        self.handle = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: Layer) c.ncnn_layer_t {
        return self.handle;
    }

    /// True when the layer consumes and produces a single blob.
    pub fn setOneBlobOnly(self: Layer, enable: bool) void {
        c.ncnn_layer_set_one_blob_only(self.handle, @intFromBool(enable));
    }

    pub fn oneBlobOnly(self: Layer) bool {
        return c.ncnn_layer_get_one_blob_only(self.handle) != 0;
    }

    /// True when the layer can modify its input in place.
    pub fn setSupportInplace(self: Layer, enable: bool) void {
        c.ncnn_layer_set_support_inplace(self.handle, @intFromBool(enable));
    }

    pub fn supportInplace(self: Layer) bool {
        return c.ncnn_layer_get_support_inplace(self.handle) != 0;
    }

    /// True when the layer understands packed layouts.
    pub fn setSupportPacking(self: Layer, enable: bool) void {
        c.ncnn_layer_set_support_packing(self.handle, @intFromBool(enable));
    }

    pub fn supportPacking(self: Layer) bool {
        return c.ncnn_layer_get_support_packing(self.handle) != 0;
    }

    /// True when the layer handles bf16 stored blobs.
    pub fn setSupportBf16Storage(self: Layer, enable: bool) void {
        c.ncnn_layer_set_support_bf16_storage(self.handle, @intFromBool(enable));
    }

    pub fn supportBf16Storage(self: Layer) bool {
        return c.ncnn_layer_get_support_bf16_storage(self.handle) != 0;
    }

    /// True when the layer handles fp16 stored blobs.
    pub fn setSupportFp16Storage(self: Layer, enable: bool) void {
        c.ncnn_layer_set_support_fp16_storage(self.handle, @intFromBool(enable));
    }

    pub fn supportFp16Storage(self: Layer) bool {
        return c.ncnn_layer_get_support_fp16_storage(self.handle) != 0;
    }

    /// Layer type name for layers created by type.
    pub fn typeName(self: Layer) ?[:0]const u8 {
        const name = c.ncnn_layer_get_type(self.handle);
        if (name == null) return null;
        return std.mem.sliceTo(name[0..], 0);
    }

    /// Layer type index for layers created by type.
    pub fn typeIndex(self: Layer) i32 {
        return c.ncnn_layer_get_typeindex(self.handle);
    }

    pub fn setLoadParam(self: Layer, callback: LoadParamFn) void {
        self.cStruct().load_param = callback;
    }

    pub fn setLoadModel(self: Layer, callback: LoadModelFn) void {
        self.cStruct().load_model = callback;
    }

    pub fn setCreatePipeline(self: Layer, callback: PipelineFn) void {
        self.cStruct().create_pipeline = callback;
    }

    pub fn setDestroyPipeline(self: Layer, callback: PipelineFn) void {
        self.cStruct().destroy_pipeline = callback;
    }

    pub fn setForward1(self: Layer, callback: Forward1Fn) void {
        self.cStruct().forward_1 = callback;
    }

    pub fn setForwardN(self: Layer, callback: ForwardNFn) void {
        self.cStruct().forward_n = callback;
    }

    pub fn setForwardInplace1(self: Layer, callback: ForwardInplace1Fn) void {
        self.cStruct().forward_inplace_1 = callback;
    }

    pub fn setForwardInplaceN(self: Layer, callback: ForwardInplaceNFn) void {
        self.cStruct().forward_inplace_n = callback;
    }

    /// Raw struct access for callbacks the wrapper does not name yet.
    pub fn cStruct(self: Layer) *c.struct___ncnn_layer_t {
        return self.handle orelse unreachable;
    }
};
