//! End-to-end check through the translate-c bindings of ncnn's C API: load a
//! parametrised network into memory, run inference and inspect the result.
//! This exercises the layer registry, the generated headers, the translated
//! bindings and the C++ runtime in one shot.

const std = @import("std");
const ncnn = @import("ncnn");

const param =
    \\7767517
    \\2 2
    \\Input            input   0 1 data
    \\ReLU             relu    1 1 data out
    \\
;

test "version is reported" {
    try std.testing.expect(ncnn.ncnn_version()[0] != 0);
}

test "relu inference" {
    const net = ncnn.ncnn_net_create() orelse return error.OutOfMemory;
    defer ncnn.ncnn_net_destroy(net);

    try std.testing.expectEqual(@as(c_int, 0), ncnn.ncnn_net_load_param_memory(net, param));

    const ex = ncnn.ncnn_extractor_create(net) orelse return error.OutOfMemory;
    defer ncnn.ncnn_extractor_destroy(ex);

    const input_data = [_]f32{ -1, 2, -3, 4 };
    const input = ncnn.ncnn_mat_create_1d(@intCast(input_data.len), null) orelse return error.OutOfMemory;
    defer ncnn.ncnn_mat_destroy(input);

    const input_ptr: [*]f32 = @ptrCast(@alignCast(ncnn.ncnn_mat_get_data(input).?));
    @memcpy(input_ptr[0..input_data.len], &input_data);

    try std.testing.expectEqual(@as(c_int, 0), ncnn.ncnn_extractor_input_index(ex, 0, input));

    var out: ncnn.ncnn_mat_t = null;
    try std.testing.expectEqual(@as(c_int, 0), ncnn.ncnn_extractor_extract_index(ex, 1, &out));
    const output = out orelse return error.ExtractFailed;
    defer ncnn.ncnn_mat_destroy(output);

    try std.testing.expectEqual(@as(c_int, 4), ncnn.ncnn_mat_get_w(output));

    const result: [*]const f32 = @ptrCast(@alignCast(ncnn.ncnn_mat_get_data(output).?));
    const expected = [_]f32{ 0, 2, 0, 4 };
    for (expected, 0..) |value, i| {
        try std.testing.expectEqual(value, result[i]);
    }
}
