//! ncnn source layout, mirrored from upstream `src/CMakeLists.txt`.
//!
//! Keep `layers` in the exact order used upstream: the order defines both
//! `layer_type_enum.h` values and the runtime layer registry indices, so
//! reordering breaks every existing `.param` file.

const std = @import("std");

/// Upstream revision this table was transcribed from.
pub const version = struct {
    pub const string = "1.0.20261020";
    pub const number: i64 = 20261020;
};

pub const Layer = struct {
    /// C++ class name, also the layer type string stored in `.param` files.
    name: []const u8,
    /// Upstream `ncnn_add_layer(X OFF)` default.
    enabled: bool = true,
};

pub const layers = [_]Layer{
    .{ .name = "AbsVal" },
    .{ .name = "ArgMax", .enabled = false },
    .{ .name = "BatchNorm" },
    .{ .name = "Bias" },
    .{ .name = "BNLL" },
    .{ .name = "Concat" },
    .{ .name = "Convolution" },
    .{ .name = "Crop" },
    .{ .name = "Deconvolution" },
    .{ .name = "Dropout" },
    .{ .name = "Eltwise" },
    .{ .name = "ELU" },
    .{ .name = "Embed" },
    .{ .name = "Exp" },
    .{ .name = "Flatten" },
    .{ .name = "InnerProduct" },
    .{ .name = "Input" },
    .{ .name = "Log" },
    .{ .name = "LRN" },
    .{ .name = "MemoryData" },
    .{ .name = "MVN" },
    .{ .name = "Pooling" },
    .{ .name = "Power" },
    .{ .name = "PReLU" },
    .{ .name = "Proposal" },
    .{ .name = "Reduction" },
    .{ .name = "ReLU" },
    .{ .name = "Reshape" },
    .{ .name = "ROIPooling" },
    .{ .name = "Scale" },
    .{ .name = "Sigmoid" },
    .{ .name = "Slice" },
    .{ .name = "Softmax" },
    .{ .name = "Split" },
    .{ .name = "SPP", .enabled = false },
    .{ .name = "TanH" },
    .{ .name = "Threshold" },
    .{ .name = "Tile" },
    .{ .name = "RNN" },
    .{ .name = "LSTM" },
    .{ .name = "BinaryOp" },
    .{ .name = "UnaryOp" },
    .{ .name = "ConvolutionDepthWise" },
    .{ .name = "Padding" },
    .{ .name = "Squeeze" },
    .{ .name = "ExpandDims" },
    .{ .name = "Normalize" },
    .{ .name = "Permute" },
    .{ .name = "PriorBox" },
    .{ .name = "DetectionOutput" },
    .{ .name = "Interp" },
    .{ .name = "DeconvolutionDepthWise" },
    .{ .name = "ShuffleChannel" },
    .{ .name = "InstanceNorm" },
    .{ .name = "Clip" },
    .{ .name = "Reorg" },
    .{ .name = "YoloDetectionOutput" },
    .{ .name = "Quantize" },
    .{ .name = "Dequantize" },
    .{ .name = "Yolov3DetectionOutput" },
    .{ .name = "PSROIPooling" },
    .{ .name = "ROIAlign" },
    .{ .name = "Packing" },
    .{ .name = "Requantize" },
    .{ .name = "Cast" },
    .{ .name = "HardSigmoid" },
    .{ .name = "SELU" },
    .{ .name = "HardSwish" },
    .{ .name = "Noop" },
    .{ .name = "PixelShuffle" },
    .{ .name = "DeepCopy" },
    .{ .name = "Mish" },
    .{ .name = "StatisticsPooling" },
    .{ .name = "Swish" },
    .{ .name = "Gemm" },
    .{ .name = "GroupNorm" },
    .{ .name = "LayerNorm" },
    .{ .name = "Softplus" },
    .{ .name = "GRU" },
    .{ .name = "MultiHeadAttention" },
    .{ .name = "GELU" },
    .{ .name = "Convolution1D" },
    .{ .name = "Pooling1D" },
    .{ .name = "ConvolutionDepthWise1D" },
    .{ .name = "Convolution3D" },
    .{ .name = "ConvolutionDepthWise3D" },
    .{ .name = "Pooling3D" },
    .{ .name = "MatMul" },
    .{ .name = "Deconvolution1D" },
    .{ .name = "DeconvolutionDepthWise1D" },
    .{ .name = "Deconvolution3D" },
    .{ .name = "DeconvolutionDepthWise3D" },
    .{ .name = "Einsum" },
    .{ .name = "DeformableConv2D" },
    .{ .name = "GLU" },
    .{ .name = "Fold" },
    .{ .name = "Unfold" },
    .{ .name = "GridSample" },
    .{ .name = "CumulativeSum" },
    .{ .name = "CopyTo" },
    .{ .name = "Erf" },
    .{ .name = "Diag" },
    .{ .name = "CELU" },
    .{ .name = "Shrink" },
    .{ .name = "RMSNorm" },
    .{ .name = "Spectrogram" },
    .{ .name = "InverseSpectrogram" },
    .{ .name = "Flip" },
    .{ .name = "SDPA" },
    .{ .name = "RotaryEmbed" },
};

/// Link-time dependencies between layer translation units. Enabling `layer`
/// also enables every entry in `needs`.
pub const dependencies = [_]struct { layer: []const u8, needs: []const []const u8 }{
    // `convolution_x86.cpp` calls the int8 GEMM utilities defined in
    // `gemm_x86.cpp`; without the Gemm TU the final link fails.
    .{ .layer = "Convolution", .needs = &.{"Gemm"} },
};

/// Always compiled, relative to the upstream `src` directory.
pub const core = [_][]const u8{
    "allocator.cpp",
    "benchmark.cpp",
    "blob.cpp",
    "c_api.cpp",
    "command.cpp",
    "cpu.cpp",
    "datareader.cpp",
    "expression.cpp",
    "gpu.cpp",
    "layer.cpp",
    "mat.cpp",
    "mat_pixel.cpp",
    "mat_pixel_affine.cpp",
    "mat_pixel_drawing.cpp",
    "mat_pixel_resize.cpp",
    "mat_pixel_rotate.cpp",
    "modelbin.cpp",
    "net.cpp",
    "option.cpp",
    "paramdict.cpp",
    "pipeline.cpp",
    "pipelinecache.cpp",
    "simpleocv.cpp",
    "simpleomp.cpp",
    "simplestl.cpp",
    "simplemath.cpp",
    "simplevk.cpp",
};

/// Linux userspace pixel formats, only available on Android upstream.
pub const android = [_][]const u8{"mat_pixel_android.cpp"};

pub fn lowerName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    return std.ascii.allocLowerString(allocator, name);
}
