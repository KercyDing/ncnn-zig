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
    try std.testing.expectEqual(@as(i32, 0), try net.inputIndex(0));
    try std.testing.expectEqual(@as(i32, 1), try net.outputIndex(0));
    try std.testing.expectError(error.IndexOutOfBounds, net.inputIndex(1));
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

test "batched mat" {
    var mat = try ncnn.Mat.create3dBatch(2, 2, 2, 3);
    defer mat.deinit();

    try std.testing.expectEqual(@as(usize, 3), mat.batch());
    try std.testing.expect(mat.nstep() > 0);

    const batch = try mat.batchFloat(0);
    try std.testing.expectEqual(mat.cstep() * mat.channels() * mat.elempack(), batch.len);
}

test "pixel roi" {
    const rgb: [4 * 4 * 3]u8 = @splat(0);

    var mat = try ncnn.pixel.fromPixelsRoi(&rgb, @backingInt(ncnn.pixel.Format.rgb), 4, 4, 12, 1, 1, 2, 2);
    defer mat.deinit();

    try std.testing.expectEqual(@as(usize, 2), mat.width());
    try std.testing.expectEqual(@as(usize, 2), mat.height());
    try std.testing.expectEqual(@as(usize, 3), mat.channels());
}

test "layer type index" {
    try std.testing.expectEqual(
        ncnn.c.ncnn_layer_type_to_index("ReLU"),
        ncnn.layer.typeToIndex("ReLU"),
    );
    try std.testing.expectEqual(@as(i32, -1), ncnn.layer.typeToIndex("NoSuchLayer"));
}

test "fromSlice3d channel padding" {
    const values = [_]f32{
        1,  2,  3,
        4,  5,  6,
        7,  8,  9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
    };

    var mat = try ncnn.Mat.fromSlice3d(3, 3, 2, &values);
    defer mat.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, try mat.channelFloat(0));
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 12, 13, 14, 15, 16, 17, 18 }, try mat.channelFloat(1));
}

test "external mat padding" {
    var tight: [18]f32 = @splat(0);
    try std.testing.expectError(error.InvalidBufferSize, ncnn.Mat.externalFloat3d(3, 3, 2, &tight));

    var padded: [24]f32 = @splat(0);
    for (&padded, 0..) |*value, i| value.* = @floatFromInt(i);

    var mat = try ncnn.Mat.externalFloat3d(3, 3, 2, &padded);
    defer mat.deinit();

    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8 }, try mat.channelFloat(0));
    try std.testing.expectEqualSlices(f32, &.{ 12, 13, 14, 15, 16, 17, 18, 19, 20 }, try mat.channelFloat(1));
}

test "pixel buffer validation" {
    const rgb: [4 * 4 * 3]u8 = @splat(0);
    const rgb_format = @backingInt(ncnn.pixel.Format.rgb);

    try std.testing.expectError(error.InvalidBufferSize, ncnn.pixel.fromPixels(rgb[0..40], rgb_format, 4, 4, 12));
    try std.testing.expectError(error.InvalidBufferSize, ncnn.pixel.fromPixels(&rgb, rgb_format, 4, 4, 8));

    var mat = try ncnn.pixel.fromPixels(&rgb, rgb_format, 4, 4, 12);
    defer mat.deinit();

    var out: [4 * 4 * 3]u8 = @splat(0);
    try std.testing.expectError(error.InvalidBufferSize, ncnn.pixel.toPixels(mat, out[0..40], rgb_format, 12));

    try std.testing.expectError(
        error.InvalidParameter,
        ncnn.pixel.fromPixelsRoi(&rgb, rgb_format, 4, 4, 12, 3, 0, 2, 2),
    );
}

test "mean normalize channel check" {
    var mat = try ncnn.Mat.fromSlice3d(1, 1, 3, &.{ 1, 2, 3 });
    defer mat.deinit();

    const short = [_]f32{0};
    try std.testing.expectError(error.InvalidParameter, ncnn.pixel.subtractMeanNormalize(mat, &short, null));

    const full = [_]f32{ 0, 0, 0 };
    try ncnn.pixel.subtractMeanNormalize(mat, &full, null);
}

test "batched mats require batchFloat" {
    var mat = try ncnn.Mat.create1dBatch(4, 2);
    defer mat.deinit();

    try std.testing.expectError(error.UnsupportedDims, mat.dataFloat());
    try std.testing.expectError(error.UnsupportedDims, mat.channelFloat(0));

    const first = try mat.batchFloat(0);
    try std.testing.expectEqual(@as(usize, 4), first.len);
}

test "invalid mat parameters" {
    try std.testing.expectError(error.InvalidParameter, ncnn.Mat.create1dElem(4, 0, 1));
    try std.testing.expectError(error.InvalidParameter, ncnn.Mat.create1dBatch(4, 0));

    var buf: [4]u8 = @splat(0);
    try std.testing.expectError(error.InvalidBufferSize, ncnn.Mat.external1dElem(8, 4, 1, &buf));
}

test "invalid pixel parameters" {
    const rgb: [4 * 4 * 3]u8 = @splat(0);
    const rgb_format = @backingInt(ncnn.pixel.Format.rgb);

    try std.testing.expectError(error.InvalidParameter, ncnn.pixel.fromPixels(&rgb, rgb_format, 0, 4, 12));

    var src = try ncnn.Mat.fromSlice1d(&.{ 1, 2, 3 });
    defer src.deinit();
    try std.testing.expectError(error.InvalidParameter, ncnn.pixel.makeBorder(src, -1, 0, 0, 0, 0, 0));

    var batched = try ncnn.Mat.create3dBatch(2, 2, 3, 2);
    defer batched.deinit();
    var out: [2 * 2 * 3]u8 = @splat(0);
    try std.testing.expectError(error.InvalidParameter, ncnn.pixel.toPixels(batched, &out, rgb_format, 6));
}

test "misaligned external mat" {
    var raw: [13]u8 = @splat(0);
    // Skip one byte so the element pointer is not f32 aligned.
    var mat = try ncnn.Mat.external1dElem(3, 4, 1, raw[1..13]);
    defer mat.deinit();

    try std.testing.expectError(error.NotFloat32, mat.dataFloat());
    try std.testing.expectError(error.NotFloat32, mat.channelFloat(0));
}

test "empty datareader fails cleanly" {
    var net = try ncnn.Net.init();
    defer net.deinit();

    // The C API installs default callbacks that read nothing.
    var reader = try ncnn.DataReader.init();
    defer reader.deinit();

    try std.testing.expectError(error.LoadParamFailed, net.loadParamDatareader(reader));
    try std.testing.expectError(error.LoadModelFailed, net.loadModelDatareader(reader));
}
