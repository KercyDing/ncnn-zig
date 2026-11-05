//! Tensor wrapper around `ncnn::Mat`.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;
const Option = @import("option.zig").Option;

/// Owns an `ncnn::Mat` allocated through the C API.
///
/// Channels may carry alignment padding, so prefer `channelFloat` over
/// assuming one contiguous buffer.
pub const Mat = struct {
    handle: c.ncnn_mat_t,

    /// 1D vector of `w` floats.
    pub fn create1d(w: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_1d(try dim(w), null));
    }

    /// 2D image of `w` x `h`.
    pub fn create2d(w: usize, h: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_2d(try dim(w), try dim(h), null));
    }

    /// 3D tensor of `w` x `h` x channels.
    pub fn create3d(w: usize, h: usize, channel_count: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_3d(try dim(w), try dim(h), try dim(channel_count), null));
    }

    /// 4D tensor of `w` x `h` x d x channels.
    pub fn create4d(w: usize, h: usize, d: usize, channel_count: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_4d(try dim(w), try dim(h), try dim(d), try dim(channel_count), null));
    }

    /// Allocates a 1D float tensor and copies `data`.
    pub fn fromSlice1d(data: []const f32) Error!Mat {
        var mat = try create1d(data.len);
        errdefer mat.deinit();
        @memcpy(mat.floatSpan(data.len), data);
        return mat;
    }

    /// Allocates a 2D float tensor and copies `data` (row-major).
    pub fn fromSlice2d(w: usize, h: usize, data: []const f32) Error!Mat {
        if (data.len != w * h) return Error.UnsupportedDims;
        var mat = try create2d(w, h);
        errdefer mat.deinit();
        @memcpy(mat.floatSpan(data.len), data);
        return mat;
    }

    /// Allocates a 3D float tensor and copies `data` (channel-major).
    pub fn fromSlice3d(w: usize, h: usize, channel_count: usize, data: []const f32) Error!Mat {
        if (data.len != w * h * channel_count) return Error.UnsupportedDims;
        var mat = try create3d(w, h, channel_count);
        errdefer mat.deinit();

        // Channels are padded to 16 bytes, so copy plane by plane.
        const plane = w * h;
        for (0..channel_count) |q| {
            const dst = try mat.channelFloat(q);
            @memcpy(dst, data[q * plane ..][0..plane]);
        }
        return mat;
    }

    /// 1D vector with explicit element size and packing.
    pub fn create1dElem(w: usize, elem_size: usize, pack: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_1d_elem(try dim(w), elem_size, try dim(pack), null));
    }

    /// 2D image with explicit element size and packing.
    pub fn create2dElem(w: usize, h: usize, elem_size: usize, pack: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_2d_elem(try dim(w), try dim(h), elem_size, try dim(pack), null));
    }

    /// 3D tensor with explicit element size and packing.
    pub fn create3dElem(w: usize, h: usize, channel_count: usize, elem_size: usize, pack: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_3d_elem(
            try dim(w),
            try dim(h),
            try dim(channel_count),
            elem_size,
            try dim(pack),
            null,
        ));
    }

    /// 4D tensor with explicit element size and packing.
    pub fn create4dElem(w: usize, h: usize, d: usize, channel_count: usize, elem_size: usize, pack: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_4d_elem(
            try dim(w),
            try dim(h),
            try dim(d),
            try dim(channel_count),
            elem_size,
            try dim(pack),
            null,
        ));
    }

    /// Wraps caller-owned float memory, which must outlive this mat.
    pub fn externalFloat1d(data: []f32) Error!Mat {
        return wrap(c.ncnn_mat_create_external_1d(try dim(data.len), @ptrCast(data.ptr), null));
    }

    /// Wraps caller-owned row-major float memory, which must outlive this mat.
    pub fn externalFloat2d(w: usize, h: usize, data: []f32) Error!Mat {
        if (data.len != w * h) return Error.UnsupportedDims;
        return wrap(c.ncnn_mat_create_external_2d(try dim(w), try dim(h), @ptrCast(data.ptr), null));
    }

    /// Wraps caller-owned float memory, which must outlive this mat.
    ///
    /// The buffer must leave room for ncnn's per-channel 16-byte alignment.
    pub fn externalFloat3d(w: usize, h: usize, channel_count: usize, data: []f32) Error!Mat {
        try checkExternalFloat3d(w, h, channel_count, data.len);
        return wrap(c.ncnn_mat_create_external_3d(
            try dim(w),
            try dim(h),
            try dim(channel_count),
            @ptrCast(data.ptr),
            null,
        ));
    }

    /// Returns a repacked copy with `elempack` channels per element.
    pub fn convertPacking(self: Mat, pack: usize) Error!Mat {
        var option = try Option.init();
        defer option.deinit();
        return self.convertPackingWithOption(pack, option);
    }

    /// Returns a repacked copy using `option`.
    pub fn convertPackingWithOption(self: Mat, pack: usize, option: Option) Error!Mat {
        var dst: c.ncnn_mat_t = null;
        const pack_c = std.math.cast(c_int, pack) orelse return Error.OutOfMemory;
        c.ncnn_convert_packing(self.handle, &dst, pack_c, option.handle);
        return wrap(dst);
    }

    /// Returns a 1D copy of the tensor data.
    pub fn flatten(self: Mat) Error!Mat {
        var option = try Option.init();
        defer option.deinit();
        return self.flattenWithOption(option);
    }

    /// Returns a 1D copy using `option`.
    pub fn flattenWithOption(self: Mat, option: Option) Error!Mat {
        var dst: c.ncnn_mat_t = null;
        c.ncnn_flatten(self.handle, &dst, option.handle);
        return wrap(dst);
    }

    /// 1D vector holding `n` batches.
    pub fn create1dBatch(w: usize, n: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_1d_batch(try dim(w), try dim(n), null));
    }

    /// 2D image holding `n` batches.
    pub fn create2dBatch(w: usize, h: usize, n: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_2d_batch(try dim(w), try dim(h), try dim(n), null));
    }

    /// 3D tensor holding `n` batches.
    pub fn create3dBatch(w: usize, h: usize, channel_count: usize, n: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_3d_batch(try dim(w), try dim(h), try dim(channel_count), try dim(n), null));
    }

    /// 4D tensor holding `n` batches.
    pub fn create4dBatch(w: usize, h: usize, d: usize, channel_count: usize, n: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_4d_batch(
            try dim(w),
            try dim(h),
            try dim(d),
            try dim(channel_count),
            try dim(n),
            null,
        ));
    }

    /// 1D batched vector with explicit element size and packing.
    pub fn create1dElemBatch(w: usize, elem_size: usize, pack: usize, n: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_1d_elem_batch(try dim(w), elem_size, try dim(pack), try dim(n), null));
    }

    /// 2D batched image with explicit element size and packing.
    pub fn create2dElemBatch(w: usize, h: usize, elem_size: usize, pack: usize, n: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_2d_elem_batch(
            try dim(w),
            try dim(h),
            elem_size,
            try dim(pack),
            try dim(n),
            null,
        ));
    }

    /// 3D batched tensor with explicit element size and packing.
    pub fn create3dElemBatch(
        w: usize,
        h: usize,
        channel_count: usize,
        elem_size: usize,
        pack: usize,
        n: usize,
    ) Error!Mat {
        return wrap(c.ncnn_mat_create_3d_elem_batch(
            try dim(w),
            try dim(h),
            try dim(channel_count),
            elem_size,
            try dim(pack),
            try dim(n),
            null,
        ));
    }

    /// 4D batched tensor with explicit element size and packing.
    pub fn create4dElemBatch(
        w: usize,
        h: usize,
        d: usize,
        channel_count: usize,
        elem_size: usize,
        pack: usize,
        n: usize,
    ) Error!Mat {
        return wrap(c.ncnn_mat_create_4d_elem_batch(
            try dim(w),
            try dim(h),
            try dim(d),
            try dim(channel_count),
            elem_size,
            try dim(pack),
            try dim(n),
            null,
        ));
    }

    /// Wraps caller-owned 4D float memory, which must outlive this mat.
    ///
    /// The buffer must leave room for ncnn's per-channel 16-byte alignment.
    pub fn externalFloat4d(w: usize, h: usize, d: usize, channel_count: usize, data: []f32) Error!Mat {
        try checkExternalBytes(w, h, d, @sizeOf(f32), channel_count, data.len * @sizeOf(f32));
        return wrap(c.ncnn_mat_create_external_4d(
            try dim(w),
            try dim(h),
            try dim(d),
            try dim(channel_count),
            @ptrCast(data.ptr),
            null,
        ));
    }

    /// Wraps caller-owned memory with explicit element size and packing.
    pub fn external1dElem(w: usize, elem_size: usize, pack: usize, data: []u8) Error!Mat {
        return wrap(c.ncnn_mat_create_external_1d_elem(
            try dim(w),
            @ptrCast(data.ptr),
            elem_size,
            try dim(pack),
            null,
        ));
    }

    /// Wraps caller-owned memory with explicit element size and packing.
    pub fn external2dElem(w: usize, h: usize, elem_size: usize, pack: usize, data: []u8) Error!Mat {
        return wrap(c.ncnn_mat_create_external_2d_elem(
            try dim(w),
            try dim(h),
            @ptrCast(data.ptr),
            elem_size,
            try dim(pack),
            null,
        ));
    }

    /// Wraps caller-owned memory with explicit element size and packing.
    ///
    /// The buffer must leave room for ncnn's per-channel 16-byte alignment.
    pub fn external3dElem(
        w: usize,
        h: usize,
        channel_count: usize,
        elem_size: usize,
        pack: usize,
        data: []u8,
    ) Error!Mat {
        try checkExternalBytes(w, h, 1, elem_size, channel_count, data.len);
        return wrap(c.ncnn_mat_create_external_3d_elem(
            try dim(w),
            try dim(h),
            try dim(channel_count),
            @ptrCast(data.ptr),
            elem_size,
            try dim(pack),
            null,
        ));
    }

    /// Wraps caller-owned 4D memory with explicit element size and packing.
    ///
    /// The buffer must leave room for ncnn's per-channel 16-byte alignment.
    pub fn external4dElem(
        w: usize,
        h: usize,
        d: usize,
        channel_count: usize,
        elem_size: usize,
        pack: usize,
        data: []u8,
    ) Error!Mat {
        try checkExternalBytes(w, h, d, elem_size, channel_count, data.len);
        return wrap(c.ncnn_mat_create_external_4d_elem(
            try dim(w),
            try dim(h),
            try dim(d),
            try dim(channel_count),
            @ptrCast(data.ptr),
            elem_size,
            try dim(pack),
            null,
        ));
    }

    /// Releases the underlying mat.
    pub fn deinit(self: *Mat) void {
        if (self.handle) |handle| c.ncnn_mat_destroy(handle);
        self.handle = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: Mat) c.ncnn_mat_t {
        return self.handle;
    }

    /// Returns an independent copy.
    pub fn clone(self: Mat) Error!Mat {
        return wrap(c.ncnn_mat_clone(self.handle, null));
    }

    /// 0 empty, 1 vector, 2 image, 3 dim, 4 batched dim.
    pub fn dims(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_dims(self.handle));
    }

    /// Vector width or image width.
    pub fn width(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_w(self.handle));
    }

    /// Image height.
    pub fn height(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_h(self.handle));
    }

    /// Depth of a 4D tensor.
    pub fn depth(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_d(self.handle));
    }

    /// Number of channels.
    pub fn channels(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_c(self.handle));
    }

    /// Batch count.
    pub fn batch(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_n(self.handle));
    }

    /// Bytes per packed element.
    pub fn elemsize(self: Mat) usize {
        return c.ncnn_mat_get_elemsize(self.handle);
    }

    /// Floats per element: 1 for plain, 4/8/16 for packed layout.
    pub fn elempack(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_elempack(self.handle));
    }

    /// Channel stride in elements.
    pub fn cstep(self: Mat) usize {
        return c.ncnn_mat_get_cstep(self.handle);
    }

    /// Batch stride in elements.
    pub fn nstep(self: Mat) usize {
        return c.ncnn_mat_get_nstep(self.handle);
    }

    /// Returns one batch as a flat f32 slice.
    pub fn batchFloat(self: Mat, batch_index: usize) Error![]f32 {
        if (!self.isFloat32()) return Error.NotFloat32;
        if (batch_index >= self.batch()) return Error.IndexOutOfBounds;

        const data = c.ncnn_mat_get_batch_data(self.handle, @intCast(batch_index)) orelse
            return Error.NotFloat32;
        const len = self.cstep() * self.channels() * self.elempack();
        const ptr: [*]f32 = @ptrCast(@alignCast(data));
        return ptr[0..len];
    }

    /// True when every element is a pack of plain f32 values.
    pub fn isFloat32(self: Mat) bool {
        const pack = self.elempack();
        return pack != 0 and self.elemsize() == pack * @sizeOf(f32);
    }

    /// Returns one channel as a flat f32 slice.
    ///
    /// Skips alignment padding between channels.
    pub fn channelFloat(self: Mat, index: usize) Error![]f32 {
        if (!self.isFloat32()) return Error.NotFloat32;
        if (index >= self.channels()) return Error.IndexOutOfBounds;

        const data = c.ncnn_mat_get_channel_data(self.handle, @intCast(index)) orelse
            return Error.NotFloat32;
        const len = self.width() * self.height() * self.depth() * self.elempack();
        const ptr: [*]f32 = @ptrCast(@alignCast(data));
        return ptr[0..len];
    }

    /// Returns the tensor as one flat f32 slice for 1D and 2D mats.
    ///
    /// 3D and 4D mats may have padded channels, use `channelFloat`.
    pub fn dataFloat(self: Mat) Error![]f32 {
        if (!self.isFloat32()) return Error.NotFloat32;

        const len: usize = switch (self.dims()) {
            1 => self.width() * self.elempack(),
            2 => self.width() * self.height() * self.elempack(),
            else => return Error.UnsupportedDims,
        };

        const data = c.ncnn_mat_get_data(self.handle) orelse return Error.NotFloat32;
        const ptr: [*]f32 = @ptrCast(@alignCast(data));
        return ptr[0..len];
    }

    /// Returns a reshaped view sharing this mat's data.
    pub fn reshape1d(self: Mat, w: usize) Error!Mat {
        return wrap(c.ncnn_mat_reshape_1d(self.handle, try dim(w), null));
    }

    /// Returns a reshaped view sharing this mat's data.
    pub fn reshape2d(self: Mat, w: usize, h: usize) Error!Mat {
        return wrap(c.ncnn_mat_reshape_2d(self.handle, try dim(w), try dim(h), null));
    }

    /// Returns a reshaped view sharing this mat's data.
    pub fn reshape3d(self: Mat, w: usize, h: usize, channel_count: usize) Error!Mat {
        return wrap(c.ncnn_mat_reshape_3d(self.handle, try dim(w), try dim(h), try dim(channel_count), null));
    }

    /// Returns a reshaped view sharing this mat's data.
    pub fn reshape4d(self: Mat, w: usize, h: usize, d: usize, channel_count: usize) Error!Mat {
        return wrap(c.ncnn_mat_reshape_4d(self.handle, try dim(w), try dim(h), try dim(d), try dim(channel_count), null));
    }

    /// Fills every element with `value`.
    pub fn fill(self: Mat, value: f32) void {
        c.ncnn_mat_fill_float(self.handle, value);
    }

    /// Raw view used by the `fromSlice*` constructors.
    fn floatSpan(self: Mat, len: usize) []f32 {
        const data = c.ncnn_mat_get_data(self.handle) orelse unreachable;
        const ptr: [*]f32 = @ptrCast(@alignCast(data));
        return ptr[0..len];
    }
};

/// Rejects null handles early.
fn wrap(handle: c.ncnn_mat_t) Error!Mat {
    return .{ .handle = handle orelse return Error.CreateMatFailed };
}

/// Rejects external buffers that do not include ncnn's channel padding.
fn checkExternalBytes(
    w: usize,
    h: usize,
    d: usize,
    elem_size: usize,
    channel_count: usize,
    buffer_bytes: usize,
) Error!void {
    const plane = std.math.mul(usize, w, h) catch return Error.InvalidBufferSize;
    const volume = std.math.mul(usize, plane, d) catch return Error.InvalidBufferSize;
    const bytes = std.math.mul(usize, volume, elem_size) catch return Error.InvalidBufferSize;
    const padded = (bytes + 15) & ~@as(usize, 15);
    const required = std.math.mul(usize, padded, channel_count) catch return Error.InvalidBufferSize;
    if (buffer_bytes < required) return Error.InvalidBufferSize;
}

fn checkExternalFloat3d(w: usize, h: usize, channel_count: usize, len: usize) Error!void {
    return checkExternalBytes(w, h, 1, @sizeOf(f32), channel_count, len * @sizeOf(f32));
}

/// ncnn sizes are `int`.
fn dim(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse Error.OutOfMemory;
}
