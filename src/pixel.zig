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
    try checkSource(pixels, format, w, h, stride);
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
    try checkSource(pixels, format, w, h, stride);
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

/// Wraps a region of interest of an image as a 3D float mat.
pub fn fromPixelsRoi(
    pixels: []const u8,
    format: i32,
    w: usize,
    h: usize,
    stride: usize,
    roi_x: i32,
    roi_y: i32,
    roi_w: i32,
    roi_h: i32,
) Error!Mat {
    try checkSource(pixels, format, w, h, stride);
    try checkRoi(w, h, roi_x, roi_y, roi_w, roi_h);
    return wrap(c.ncnn_mat_from_pixels_roi(
        pixels.ptr,
        format,
        try dim(w),
        try dim(h),
        try dim(stride),
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        null,
    ));
}

/// Wraps and resizes a region of interest of an image.
pub fn fromPixelsRoiResize(
    pixels: []const u8,
    format: i32,
    w: usize,
    h: usize,
    stride: usize,
    roi_x: i32,
    roi_y: i32,
    roi_w: i32,
    roi_h: i32,
    target_w: usize,
    target_h: usize,
) Error!Mat {
    try checkSource(pixels, format, w, h, stride);
    try checkRoi(w, h, roi_x, roi_y, roi_w, roi_h);
    return wrap(c.ncnn_mat_from_pixels_roi_resize(
        pixels.ptr,
        format,
        try dim(w),
        try dim(h),
        try dim(stride),
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        try dim(target_w),
        try dim(target_h),
        null,
    ));
}

/// Writes a float mat back into a tightly packed 8-bit image.
pub fn toPixels(mat: Mat, pixels: []u8, format: i32, stride: usize) Error!void {
    try checkDestination(mat, pixels, format, mat.width(), mat.height(), stride);
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
    try checkDestination(mat, pixels, format, target_w, target_h, target_stride);
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
pub fn subtractMeanNormalize(mat: Mat, mean: ?[]const f32, norm: ?[]const f32) Error!void {
    if (mean) |values| {
        if (values.len < mat.channels()) return Error.InvalidParameter;
    }
    if (norm) |values| {
        if (values.len < mat.channels()) return Error.InvalidParameter;
    }

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

/// Returns a 3D bordered copy of `src`, padding channels with `front`/`behind`.
pub fn makeBorder3d(
    src: Mat,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    front: i32,
    behind: i32,
    border_type: i32,
    value: f32,
) Error!Mat {
    const dst = c.ncnn_mat_create() orelse return Error.CreateMatFailed;
    c.ncnn_copy_make_border_3d(src.handle, dst, top, bottom, left, right, front, behind, border_type, value, null);
    return .{ .handle = dst };
}

/// Returns a 3D cropped copy of `src`.
pub fn cutBorder3d(
    src: Mat,
    top: i32,
    bottom: i32,
    left: i32,
    right: i32,
    front: i32,
    behind: i32,
) Error!Mat {
    const dst = c.ncnn_mat_create() orelse return Error.CreateMatFailed;
    c.ncnn_copy_cut_border_3d(src.handle, dst, top, bottom, left, right, front, behind, null);
    return .{ .handle = dst };
}

/// Channels of the source side of a pixel type.
fn sourceChannels(format: i32) Error!usize {
    const base = format & 0x0000ffff;
    return switch (base) {
        c.NCNN_MAT_PIXEL_RGB, c.NCNN_MAT_PIXEL_BGR => 3,
        c.NCNN_MAT_PIXEL_GRAY => 1,
        c.NCNN_MAT_PIXEL_RGBA, c.NCNN_MAT_PIXEL_BGRA => 4,
        else => Error.UnsupportedPixelFormat,
    };
}

/// Channels of the destination side of a pixel type.
fn destinationChannels(format: i32) Error!usize {
    const converted = format & ~@as(i32, 0x0000ffff);
    if (converted != 0) return sourceChannels(format >> 16);
    return sourceChannels(format);
}

/// Validates an input image and its stride.
fn checkSource(pixels: []const u8, format: i32, w: usize, h: usize, stride: usize) Error!void {
    const channels = try sourceChannels(format);
    const required = try imageBytes(w, h, channels, stride);
    if (pixels.len < required) return Error.InvalidBufferSize;
}

/// Validates mat shape, output image and its stride.
fn checkDestination(
    mat: Mat,
    pixels: []const u8,
    format: i32,
    target_w: usize,
    target_h: usize,
    stride: usize,
) Error!void {
    if (mat.dims() != 3) return Error.InvalidParameter;
    if (mat.channels() != try sourceChannels(format)) return Error.InvalidParameter;

    const channels = try destinationChannels(format);
    const required = try imageBytes(target_w, target_h, channels, stride);
    if (pixels.len < required) return Error.InvalidBufferSize;
}

/// Minimum buffer size for `h` rows of `w * channels` bytes at `stride`.
fn imageBytes(w: usize, h: usize, channels: usize, stride: usize) Error!usize {
    if (w == 0 or h == 0) return 0;

    const row = std.math.mul(usize, w, channels) catch return Error.InvalidBufferSize;
    if (stride < row) return Error.InvalidBufferSize;

    const rows = std.math.mul(usize, h - 1, stride) catch return Error.InvalidBufferSize;
    return std.math.add(usize, rows, row) catch return Error.InvalidBufferSize;
}

/// Rejects regions outside the source image.
fn checkRoi(w: usize, h: usize, roi_x: i32, roi_y: i32, roi_w: i32, roi_h: i32) Error!void {
    if (roi_x < 0 or roi_y < 0 or roi_w <= 0 or roi_h <= 0) return Error.InvalidParameter;

    const x: usize = @intCast(roi_x);
    const y: usize = @intCast(roi_y);
    const rw: usize = @intCast(roi_w);
    const rh: usize = @intCast(roi_h);

    if (x + rw > w or y + rh > h) return Error.InvalidParameter;
}

/// Rejects null handles early.
fn wrap(handle: c.ncnn_mat_t) Error!Mat {
    return .{ .handle = handle orelse return Error.CreateMatFailed };
}

/// ncnn sizes are `int`.
fn dim(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse Error.OutOfMemory;
}
