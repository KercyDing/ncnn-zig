//! Pixel format conversion and border helpers.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;
const Mat = @import("mat.zig").Mat;

/// Pixel layouts accepted by the pixel helpers.
pub const Format = enum(c_int) {
    rgb = c.NCNN_MAT_PIXEL_RGB,
    bgr = c.NCNN_MAT_PIXEL_BGR,
    gray = c.NCNN_MAT_PIXEL_GRAY,
    rgba = c.NCNN_MAT_PIXEL_RGBA,
    bgra = c.NCNN_MAT_PIXEL_BGRA,

    /// Packs a source and destination layout into an `x2y` conversion type.
    pub fn convert(src: Format, dst: Format) i32 {
        return @backingInt(src) | (@as(i32, @backingInt(dst)) << 16);
    }
};

/// Wraps a tightly packed 8-bit image as a 3D float mat.
pub fn fromPixels(pixels: []const u8, format: i32, w: usize, h: usize, stride: usize) Error!Mat {
    return wrap(c.ncnn_mat_from_pixels(pixels.ptr, format, try dim(w), try dim(h), try dim(stride), null));
}

/// Wraps an image and resizes it while converting to float.
pub fn fromPixelsResize(
    pixels: []const u8,
    format: i32,
    w: usize,
    h: usize,
    stride: usize,
    target_w: usize,
    target_h: usize,
) Error!Mat {
    return wrap(c.ncnn_mat_from_pixels_resize(
        pixels.ptr,
        format,
        try dim(w),
        try dim(h),
        try dim(stride),
        try dim(target_w),
        try dim(target_h),
        null,
    ));
}

/// Writes a float mat back into a tightly packed 8-bit image.
pub fn toPixels(mat: Mat, pixels: []u8, format: i32, stride: usize) Error!void {
    c.ncnn_mat_to_pixels(mat.handle, pixels.ptr, format, try dim(stride));
}

/// Writes a resized copy of a float mat into an 8-bit image.
pub fn toPixelsResize(
    mat: Mat,
    pixels: []u8,
    format: i32,
    target_w: usize,
    target_h: usize,
    target_stride: usize,
) Error!void {
    c.ncnn_mat_to_pixels_resize(
        mat.handle,
        pixels.ptr,
        format,
        try dim(target_w),
        try dim(target_h),
        try dim(target_stride),
    );
}

/// Applies per-channel mean subtraction and normalization in place.
pub fn subtractMeanNormalize(mat: Mat, mean: ?[]const f32, norm: ?[]const f32) void {
    c.ncnn_mat_substract_mean_normalize(
        mat.handle,
        if (mean) |values| values.ptr else null,
        if (norm) |values| values.ptr else null,
    );
}

/// Returns a bordered copy of `src`.
pub fn makeBorder(
    src: Mat,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    border_type: i32,
    value: f32,
) Error!Mat {
    const dst = c.ncnn_mat_create() orelse return Error.CreateMatFailed;
    c.ncnn_copy_make_border(src.handle, dst, top, bottom, left, right, border_type, value, null);
    return .{ .handle = dst };
}

/// Returns a cropped copy of `src`.
pub fn cutBorder(src: Mat, top: i32, bottom: i32, left: i32, right: i32) Error!Mat {
    const dst = c.ncnn_mat_create() orelse return Error.CreateMatFailed;
    c.ncnn_copy_cut_border(src.handle, dst, top, bottom, left, right, null);
    return .{ .handle = dst };
}

/// Rejects null handles early.
fn wrap(handle: c.ncnn_mat_t) Error!Mat {
    return .{ .handle = handle orelse return Error.CreateMatFailed };
}

/// ncnn sizes are `int`.
fn dim(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse Error.OutOfMemory;
}
