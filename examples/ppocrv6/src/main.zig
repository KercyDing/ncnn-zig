//! PP-OCRv6 text-line recognition with ncnn.
//!
//! This example intentionally covers only the recognition model: it runs a
//! built-in sample text line, not a full page. Detection is left out so the
//! example stays small and focused on exercising the ncnn.zig bindings.

const std = @import("std");
const ncnn = @import("ncnn");
const zignal = @import("zignal");

const Rgb = zignal.Rgb(u8);

const model_param = @embedFile("models/PP_OCRv6_tiny_rec.param");
const model_bin = @embedFile("models/PP_OCRv6_tiny_rec.bin");
const dictionary = @embedFile("models/ppocr_keys_v6_tiny.txt");
const sample_image = @embedFile("assets/sample.png");
const target_h = 48;
const means = [_]f32{ 127.5, 127.5, 127.5 };
const norms = [_]f32{ 1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5 };

pub fn main(init: std.process.Init) !void {
    var image: zignal.Image(Rgb) = try .loadFromBytes(init.io, init.gpa, sample_image);
    defer image.deinit(init.gpa);

    const width: usize = image.cols;
    const height: usize = image.rows;
    const ratio = @as(f32, @floatFromInt(target_h)) / @as(f32, @floatFromInt(height));
    const resized_w: usize = @max(1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(width)) * ratio)));

    const stride = image.stride * @sizeOf(Rgb);
    const pixels: []const u8 = @as([*]const u8, @ptrCast(image.data.ptr))[0 .. image.data.len * @sizeOf(Rgb)];

    var blob = try ncnn.pixel.fromPixelsResize(
        pixels,
        @backingInt(ncnn.pixel.Format.rgb),
        width,
        height,
        stride,
        resized_w,
        target_h,
    );
    defer blob.deinit();
    try ncnn.pixel.subtractMeanNormalize(blob, &means, &norms);

    var net = try ncnn.Net.init();
    defer net.deinit();
    net.setNumThreads(1);
    try net.loadParamMemory(model_param[0..]);
    _ = try net.loadModelMemory(model_bin[0..]);

    var extractor = try net.createExtractor();
    defer extractor.deinit();
    try extractor.input("input", blob);
    var output = try extractor.extract("output");
    defer output.deinit();

    const rows = output.height();
    const cols = output.width();
    std.debug.print("output: {d}x{d}\n", .{ rows, cols });

    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(init.gpa);
    // `getline` semantics: a trailing newline does not create an extra key,
    // but empty lines inside the dictionary are real (blank) entries.
    const keys_body = if (dictionary.len != 0 and dictionary[dictionary.len - 1] == '\n')
        dictionary[0 .. dictionary.len - 1]
    else
        dictionary;
    var lines = std.mem.splitScalar(u8, keys_body, '\n');
    while (lines.next()) |line| {
        try keys.append(init.gpa, std.mem.trimEnd(u8, line, "\r"));
    }

    if (keys.items.len != cols) {
        std.debug.print("keys count {d} does not match output columns {d}\n", .{ keys.items.len, cols });
        return error.KeyCountMismatch;
    }

    const scores = try output.dataFloat();
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(init.gpa);

    var total_score: f32 = 0;
    var chars: usize = 0;
    var previous: usize = 0; // blank
    for (0..rows) |row| {
        const row_scores = scores[row * cols ..][0..cols];
        var best_index: usize = 0;
        var best_score: f32 = row_scores[0];
        for (row_scores[1..], 1..) |score, index| {
            if (score > best_score) {
                best_score = score;
                best_index = index;
            }
        }
        if (best_index != 0 and best_index != previous) {
            try text.appendSlice(init.gpa, keys.items[best_index]);
            total_score += best_score;
            chars += 1;
        }
        previous = best_index;
    }

    const trimmed = std.mem.trim(u8, text.items, " \t\r\n");
    std.debug.print("text: {s}\n", .{trimmed});
    if (chars != 0) {
        std.debug.print("score: {d:.4}\n", .{total_score / @as(f32, @floatFromInt(chars))});
    }
}
