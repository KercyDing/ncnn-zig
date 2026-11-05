//! Pixel drawing helpers for tightly packed 8-bit images.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// Draws a rectangle; `thickness` -1 fills it.
pub fn rectangle(
    pixels: []u8,
    w: usize,
    h: usize,
    channels: usize,
    x: i32,
    y: i32,
    rect_w: i32,
    rect_h: i32,
    color: u32,
    thickness: i32,
) Error!void {
    try check(pixels, w, h, channels);
    const pw = try dim(w);
    const ph = try dim(h);

    switch (channels) {
        1 => c.ncnn_draw_rectangle_c1(pixels.ptr, pw, ph, x, y, rect_w, rect_h, color, thickness),
        2 => c.ncnn_draw_rectangle_c2(pixels.ptr, pw, ph, x, y, rect_w, rect_h, color, thickness),
        3 => c.ncnn_draw_rectangle_c3(pixels.ptr, pw, ph, x, y, rect_w, rect_h, color, thickness),
        4 => c.ncnn_draw_rectangle_c4(pixels.ptr, pw, ph, x, y, rect_w, rect_h, color, thickness),
        else => unreachable,
    }
}

/// Draws a circle outline; `thickness` -1 fills it.
pub fn circle(
    pixels: []u8,
    w: usize,
    h: usize,
    channels: usize,
    cx: i32,
    cy: i32,
    radius: i32,
    color: u32,
    thickness: i32,
) Error!void {
    try check(pixels, w, h, channels);
    const pw = try dim(w);
    const ph = try dim(h);

    switch (channels) {
        1 => c.ncnn_draw_circle_c1(pixels.ptr, pw, ph, cx, cy, radius, color, thickness),
        2 => c.ncnn_draw_circle_c2(pixels.ptr, pw, ph, cx, cy, radius, color, thickness),
        3 => c.ncnn_draw_circle_c3(pixels.ptr, pw, ph, cx, cy, radius, color, thickness),
        4 => c.ncnn_draw_circle_c4(pixels.ptr, pw, ph, cx, cy, radius, color, thickness),
        else => unreachable,
    }
}

/// Draws a line.
pub fn line(
    pixels: []u8,
    w: usize,
    h: usize,
    channels: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: u32,
    thickness: i32,
) Error!void {
    try check(pixels, w, h, channels);
    const pw = try dim(w);
    const ph = try dim(h);

    switch (channels) {
        1 => c.ncnn_draw_line_c1(pixels.ptr, pw, ph, x0, y0, x1, y1, color, thickness),
        2 => c.ncnn_draw_line_c2(pixels.ptr, pw, ph, x0, y0, x1, y1, color, thickness),
        3 => c.ncnn_draw_line_c3(pixels.ptr, pw, ph, x0, y0, x1, y1, color, thickness),
        4 => c.ncnn_draw_line_c4(pixels.ptr, pw, ph, x0, y0, x1, y1, color, thickness),
        else => unreachable,
    }
}

/// Draws `text` with ncnn's built-in font.
pub fn text(
    pixels: []u8,
    w: usize,
    h: usize,
    channels: usize,
    string: [:0]const u8,
    x: i32,
    y: i32,
    font_pixel_size: i32,
    color: u32,
) Error!void {
    try check(pixels, w, h, channels);
    const pw = try dim(w);
    const ph = try dim(h);

    switch (channels) {
        1 => c.ncnn_draw_text_c1(pixels.ptr, pw, ph, string.ptr, x, y, font_pixel_size, color),
        2 => c.ncnn_draw_text_c2(pixels.ptr, pw, ph, string.ptr, x, y, font_pixel_size, color),
        3 => c.ncnn_draw_text_c3(pixels.ptr, pw, ph, string.ptr, x, y, font_pixel_size, color),
        4 => c.ncnn_draw_text_c4(pixels.ptr, pw, ph, string.ptr, x, y, font_pixel_size, color),
        else => unreachable,
    }
}

/// Rejects buffers that cannot hold `w * h * channels` bytes.
fn check(pixels: []u8, w: usize, h: usize, channels: usize) Error!void {
    if (channels < 1 or channels > 4) return Error.UnsupportedDims;
    if (pixels.len < w * h * channels) return Error.InvalidPixelBuffer;
}

/// ncnn sizes are `int`.
fn dim(value: usize) Error!c_int {
    return std.math.cast(c_int, value) orelse Error.OutOfMemory;
}
