//! End-to-end checks for the Zig wrapper.

const std = @import("std");
const ncnn = @import("ncnn");

// Minimal Input -> ReLU network, no weights needed.
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

test "paramdict round trip" {
    var pd = try ncnn.ParamDict.init();
    defer pd.deinit();

    pd.setInt(0, 7);
    pd.setFloat(1, 0.5);

    try std.testing.expectEqual(@as(i32, 7), pd.getInt(0, 0));
    try std.testing.expectEqual(@as(f32, 0.5), pd.getFloat(1, 0.0));
    try std.testing.expectEqual(@as(i32, 9), pd.getInt(2, 9));
}

test "mat reshape" {
    var flat = try ncnn.Mat.fromSlice1d(&.{ 1, 2, 3, 4, 5, 6 });
    defer flat.deinit();

    var image = try flat.reshape2d(3, 2);
    defer image.deinit();

    try std.testing.expectEqual(@as(usize, 2), image.dims());
    try std.testing.expectEqual(@as(usize, 3), image.width());
    try std.testing.expectEqual(@as(usize, 2), image.height());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try image.dataFloat());
}

test "pixel round trip" {
    const rgba = [_]u8{ 255, 0, 0, 128, 0, 255, 0, 64 };

    var mat = try ncnn.pixel.fromPixels(&rgba, @backingInt(ncnn.pixel.Format.rgba), 2, 1, 8);
    defer mat.deinit();

    try std.testing.expectEqual(@as(usize, 3), mat.dims());
    try std.testing.expectEqual(@as(usize, 2), mat.width());
    try std.testing.expectEqual(@as(usize, 4), mat.channels());

    var out: [rgba.len]u8 = @splat(0);
    try ncnn.pixel.toPixels(mat, &out, @backingInt(ncnn.pixel.Format.rgba), 8);
    try std.testing.expectEqualSlices(u8, &rgba, &out);
}

test "draw rectangle" {
    var pixels: [4 * 4 * 3]u8 = @splat(0);
    try ncnn.draw.rectangle(&pixels, 4, 4, 3, 0, 0, 4, 4, 0x00ff00, -1);

    for (pixels, 0..) |value, i| {
        if (i % 3 == 1) {
            try std.testing.expectEqual(@as(u8, 0xff), value);
        }
    }
}

test "option quantized switches" {
    var opt = try ncnn.Option.init();
    defer opt.deinit();

    opt.setUseFp16Arithmetic(true);
    try std.testing.expect(opt.useFp16Arithmetic());

    opt.setUseInt8Storage(true);
    try std.testing.expect(opt.useInt8Storage());

    opt.setUseBf16Packed(true);
    try std.testing.expect(opt.useBf16Packed());
}

test "external mat" {
    var data = [_]f32{ 1, 2, 3 };
    var mat = try ncnn.Mat.externalFloat1d(&data);
    defer mat.deinit();

    (try mat.dataFloat())[0] = 9;
    try std.testing.expectEqual(@as(f32, 9), data[0]);
}

test "packing and flatten" {
    const values = [_]f32{
        1, 2,  3,  4,  5,  6,  7,  8,
        9, 10, 11, 12, 13, 14, 15, 16,
    };

    var mat = try ncnn.Mat.fromSlice3d(2, 2, 4, &values);
    defer mat.deinit();

    var repacked = try mat.convertPacking(4);
    defer repacked.deinit();
    try std.testing.expectEqual(@as(usize, 4), repacked.elempack());
    try std.testing.expectEqual(@as(usize, 1), repacked.channels());

    var flat = try mat.flatten();
    defer flat.deinit();
    try std.testing.expectEqual(@as(usize, 1), flat.dims());
    try std.testing.expectEqual(@as(usize, 16), (try flat.dataFloat()).len);
}

fn addHundredInplace(
    handle: ncnn.c.ncnn_layer_t,
    blob: ncnn.c.ncnn_mat_t,
    opt: ncnn.c.ncnn_option_t,
) callconv(.c) c_int {
    _ = handle;
    _ = opt;

    const mat = ncnn.Mat{ .handle = blob };
    const data = mat.dataFloat() catch return -1;
    for (data) |*value| value.* += 100;
    return 0;
}

fn createAddHundred(userdata: ?*anyopaque) callconv(.c) ncnn.c.ncnn_layer_t {
    _ = userdata;

    var layer = ncnn.Layer.init() catch return null;
    layer.setOneBlobOnly(true);
    layer.setSupportInplace(true);
    layer.setForwardInplace1(addHundredInplace);
    return layer.raw();
}

fn destroyAddHundred(handle: ncnn.c.ncnn_layer_t, userdata: ?*anyopaque) callconv(.c) void {
    _ = userdata;
    var layer = ncnn.Layer{ .handle = handle };
    layer.deinit();
}

test "custom layer" {
    var net = try ncnn.Net.init();
    defer net.deinit();

    try net.registerCustomLayerByType("AddHundred", createAddHundred, destroyAddHundred, null);
    try net.loadParamMemory(
        \\7767517
        \\2 2
        \\Input            input   0 1 data
        \\AddHundred       add     1 1 data out
    );

    var extractor = try net.createExtractor();
    defer extractor.deinit();

    var input = try ncnn.Mat.fromSlice1d(&.{ 1, 2, 3 });
    defer input.deinit();
    try extractor.input("data", input);

    var output = try extractor.extract("out");
    defer output.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 101, 102, 103 }, try output.dataFloat());
}

test "datareader param" {
    var net = try ncnn.Net.init();
    defer net.deinit();

    var reader = try ncnn.DataReader.fromMemory(param);
    defer reader.deinit();

    try net.loadParamDatareader(reader);
    try std.testing.expectEqual(@as(usize, 1), net.inputCount());
    try std.testing.expectEqual(@as(i32, 0), net.inputIndex(0));
    try std.testing.expectEqual(@as(i32, 1), net.outputIndex(0));
}

test "allocator option" {
    var allocator = try ncnn.Allocator.initPool();
    defer allocator.deinit();

    var option = try ncnn.Option.init();
    defer option.deinit();
    option.setBlobAllocator(allocator);
    option.setWorkspaceAllocator(allocator);

    var net = try ncnn.Net.init();
    defer net.deinit();
    net.setOption(option);
    try net.loadParamMemory(param);

    var extractor = try net.createExtractor();
    defer extractor.deinit();
    extractor.setOption(option);

    var input = try ncnn.Mat.fromSlice1d(&.{ -1, 2 });
    defer input.deinit();
    try extractor.input("data", input);

    var output = try extractor.extract("out");
    defer output.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 2 }, try output.dataFloat());
}

test "3d border" {
    const values = [_]f32{
        1, 2, 3, 4,
        5, 6, 7, 8,
    };

    var mat = try ncnn.Mat.fromSlice3d(2, 2, 2, &values);
    defer mat.deinit();

    var bordered = try ncnn.pixel.makeBorder3d(mat, 1, 1, 1, 1, 1, 1, 0, 0.0);
    defer bordered.deinit();

    try std.testing.expectEqual(@as(usize, 4), bordered.width());
    try std.testing.expectEqual(@as(usize, 4), bordered.height());
    try std.testing.expectEqual(@as(usize, 4), bordered.channels());
}

test "layer by type" {
    var layer = try ncnn.Layer.createByType("ReLU");
    defer layer.deinit();

    // ReLU enables both in its constructor.
    try std.testing.expect(layer.oneBlobOnly());
    try std.testing.expect(layer.supportInplace());
}
