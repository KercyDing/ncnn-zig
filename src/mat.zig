const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// Owns an `ncnn::Mat` allocated through the C API.
///
/// ncnn stores channel-aligned data: use `channelFloat` for tensors instead
/// of assuming one flat contiguous buffer.
pub const Mat = struct {
    handle: c.ncnn_mat_t,

    pub fn create1d(w: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_1d(try dim(w), null));
    }

    pub fn create2d(w: usize, h: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_2d(try dim(w), try dim(h), null));
    }

    pub fn create3d(w: usize, h: usize, channel_count: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_3d(try dim(w), try dim(h), try dim(channel_count), null));
    }

    pub fn create4d(w: usize, h: usize, d: usize, channel_count: usize) Error!Mat {
        return wrap(c.ncnn_mat_create_4d(try dim(w), try dim(h), try dim(d), try dim(channel_count), null));
    }

    /// Allocates a 1D float tensor and copies `data` into it.
    pub fn fromSlice1d(data: []const f32) Error!Mat {
        var mat = try create1d(data.len);
        errdefer mat.deinit();
        @memcpy(mat.floatSpan(data.len), data);
        return mat;
    }

    /// Allocates a 2D float tensor and copies `data` (row-major, `w * h`).
    pub fn fromSlice2d(w: usize, h: usize, data: []const f32) Error!Mat {
        if (data.len != w * h) return Error.UnsupportedDims;
        var mat = try create2d(w, h);
        errdefer mat.deinit();
        @memcpy(mat.floatSpan(data.len), data);
        return mat;
    }

    /// Allocates a 3D float tensor and copies `data` (channel-major, `w * h * c`).
    pub fn fromSlice3d(w: usize, h: usize, channel_count: usize, data: []const f32) Error!Mat {
        if (data.len != w * h * channel_count) return Error.UnsupportedDims;
        var mat = try create3d(w, h, channel_count);
        errdefer mat.deinit();
        @memcpy(mat.floatSpan(data.len), data);
        return mat;
    }

    pub fn deinit(self: *Mat) void {
        if (self.handle) |handle| c.ncnn_mat_destroy(handle);
        self.handle = null;
    }

    pub fn raw(self: Mat) c.ncnn_mat_t {
        return self.handle;
    }

    pub fn clone(self: Mat) Error!Mat {
        return wrap(c.ncnn_mat_clone(self.handle, null));
    }

    pub fn dims(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_dims(self.handle));
    }

    pub fn width(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_w(self.handle));
    }

    pub fn height(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_h(self.handle));
    }

    pub fn depth(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_d(self.handle));
    }

    pub fn channels(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_c(self.handle));
    }

    pub fn batch(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_n(self.handle));
    }

    pub fn elemsize(self: Mat) usize {
        return c.ncnn_mat_get_elemsize(self.handle);
    }

    pub fn elempack(self: Mat) usize {
        return @intCast(c.ncnn_mat_get_elempack(self.handle));
    }

    pub fn cstep(self: Mat) usize {
        return c.ncnn_mat_get_cstep(self.handle);
    }

    /// True when every element is a pack of plain `f32` values.
    pub fn isFloat32(self: Mat) bool {
        const pack = self.elempack();
        return pack != 0 and self.elemsize() == pack * @sizeOf(f32);
    }

    /// Returns one channel as a flat `f32` slice.
    ///
    /// The length is `width * height * depth * elempack`; padding between
    /// channels is excluded.
    pub fn channelFloat(self: Mat, index: usize) Error![]f32 {
        if (!self.isFloat32()) return Error.NotFloat32;
        if (index >= self.channels()) return Error.IndexOutOfBounds;

        const data = c.ncnn_mat_get_channel_data(self.handle, @intCast(index)) orelse
            return Error.NotFloat32;
        const len = self.width() * self.height() * self.depth() * self.elempack();
        const ptr: [*]f32 = @ptrCast(@alignCast(data));
        return ptr[0..len];
    }

    /// Returns the whole tensor as a flat `f32` slice for 1D and 2D mats.
    ///
    /// 3D and 4D mats may have channel padding, so use `channelFloat`.
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

    pub fn fill(self: Mat, value: f32) void {
        c.ncnn_mat_fill_float(self.handle, value);
    }

    fn floatSpan(self: Mat, len: usize) []f32 {
        const data = c.ncnn_mat_get_data(self.handle) orelse unreachable;
        const ptr: [*]f32 = @ptrCast(@alignCast(data));
        return ptr[0..len];
    }
};

fn wrap(handle: c.ncnn_mat_t) Error!Mat {
    return .{ .handle = handle orelse return Error.CreateMatFailed };
}

fn dim(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse Error.OutOfMemory;
}
