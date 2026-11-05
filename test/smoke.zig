const std = @import("std");
const ncnn = @import("ncnn");

const param =
    \\7767517
    \\2 2
    \\Input            input   0 1 data
    \\ReLU             relu    1 1 data out
;

test "version matches headers" {
    try ncnn.checkVersion();
    try std.testing.expect(ncnn.version.string().len > 0);
}

test "option round trips" {
    var opt = try ncnn.Option.init();
    defer opt.deinit();

    opt.setNumThreads(2);
    try std.testing.expectEqual(@as(i32, 2), opt.numThreads());

    opt.setUsePackingLayout(false);
    try std.testing.expect(!opt.usePackingLayout());
}

test "relu inference" {
    var net = try ncnn.Net.init();
    defer net.deinit();

    try net.loadParamMemory(param);
    try std.testing.expectEqual(@as(usize, 1), net.inputCount());
    try std.testing.expectEqualStrings("data", net.inputName(0).?);

    var extractor = try net.createExtractor();
    defer extractor.deinit();

    var input = try ncnn.Mat.fromSlice1d(&.{ -1, 2, -3, 4 });
    defer input.deinit();
    try extractor.input("data", input);

    var output = try extractor.extract("out");
    defer output.deinit();

    try std.testing.expectEqual(@as(usize, 4), output.width());
    const values = try output.dataFloat();
    try std.testing.expectEqualSlices(f32, &.{ 0, 2, 0, 4 }, values);
}

test "mat channels" {
    var mat = try ncnn.Mat.fromSlice3d(2, 2, 2, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer mat.deinit();

    try std.testing.expectEqual(@as(usize, 3), mat.dims());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try mat.channelFloat(0));
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 7, 8 }, try mat.channelFloat(1));
    try std.testing.expectError(error.UnsupportedDims, mat.dataFloat());
    try std.testing.expectError(error.IndexOutOfBounds, mat.channelFloat(2));
}
