const std = @import("std");
const ncnn = @import("ncnn");

const param =
    \\7767517
    \\2 2
    \\Input            input   0 1 data
    \\ReLU             relu    1 1 data out
;

pub fn main() !void {
    var net = try ncnn.Net.init();
    defer net.deinit();
    try net.loadParamMemory(param);

    var extractor = try net.createExtractor();
    defer extractor.deinit();

    var input = try ncnn.Mat.fromSlice1d(&.{ -1, 2, -3, 4 });
    defer input.deinit();
    try extractor.input("data", input);

    var output = try extractor.extract("out");
    defer output.deinit();

    const values = try output.dataFloat();
    std.debug.print("ncnn {s}: {any}\n", .{ ncnn.version.string(), values });
}
