//! Inference settings wrapper around `ncnn::Option`.

const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// Owns an `ncnn::Option`.
///
/// `Net` and `Extractor` copy the option when it is applied, so the caller
/// keeps ownership and must call `deinit`.
pub const Option = struct {
    handle: c.ncnn_option_t,

    /// Creates an option with ncnn defaults.
    pub fn init() Error!Option {
        return .{ .handle = c.ncnn_option_create() orelse return Error.CreateOptionFailed };
    }

    /// Releases the option.
    pub fn deinit(self: *Option) void {
        if (self.handle) |handle| c.ncnn_option_destroy(handle);
        self.handle = null;
    }

    /// Returns the underlying C handle.
    pub fn raw(self: Option) c.ncnn_option_t {
        return self.handle;
    }

    /// Worker threads per inference, defaults to the CPU count.
    pub fn setNumThreads(self: Option, num_threads: i32) void {
        c.ncnn_option_set_num_threads(self.handle, num_threads);
    }

    pub fn numThreads(self: Option) i32 {
        return c.ncnn_option_get_num_threads(self.handle);
    }

    /// Packs 4/8/16 channels together for SIMD kernels.
    pub fn setUsePackingLayout(self: Option, enable: bool) void {
        c.ncnn_option_set_use_packing_layout(self.handle, @intFromBool(enable));
    }

    pub fn usePackingLayout(self: Option) bool {
        return c.ncnn_option_get_use_packing_layout(self.handle) != 0;
    }

    /// Picks winograd kernels for 3x3 convolution.
    pub fn setUseWinogradConvolution(self: Option, enable: bool) void {
        c.ncnn_option_set_use_winograd_convolution(self.handle, @intFromBool(enable));
    }

    pub fn useWinogradConvolution(self: Option) bool {
        return c.ncnn_option_get_use_winograd_convolution(self.handle) != 0;
    }

    /// Picks sgemm kernels for 1x1 convolution.
    pub fn setUseSgemmConvolution(self: Option, enable: bool) void {
        c.ncnn_option_set_use_sgemm_convolution(self.handle, @intFromBool(enable));
    }

    pub fn useSgemmConvolution(self: Option) bool {
        return c.ncnn_option_get_use_sgemm_convolution(self.handle) != 0;
    }

    /// Keeps intermediate blobs as fp16.
    pub fn setUseFp16Storage(self: Option, enable: bool) void {
        c.ncnn_option_set_use_fp16_storage(self.handle, @intFromBool(enable));
    }

    pub fn useFp16Storage(self: Option) bool {
        return c.ncnn_option_get_use_fp16_storage(self.handle) != 0;
    }

    /// Keeps intermediate blobs as bf16.
    pub fn setUseBf16Storage(self: Option, enable: bool) void {
        c.ncnn_option_set_use_bf16_storage(self.handle, @intFromBool(enable));
    }

    pub fn useBf16Storage(self: Option) bool {
        return c.ncnn_option_get_use_bf16_storage(self.handle) != 0;
    }

    /// Reuses a per-extractor pool instead of the global one.
    pub fn setUseLocalPoolAllocator(self: Option, enable: bool) void {
        c.ncnn_option_set_use_local_pool_allocator(self.handle, @intFromBool(enable));
    }

    pub fn useLocalPoolAllocator(self: Option) bool {
        return c.ncnn_option_get_use_local_pool_allocator(self.handle) != 0;
    }

    /// Off in this package: the library is built without Vulkan.
    pub fn setUseVulkanCompute(self: Option, enable: bool) void {
        c.ncnn_option_set_use_vulkan_compute(self.handle, @intFromBool(enable));
    }

    pub fn useVulkanCompute(self: Option) bool {
        return c.ncnn_option_get_use_vulkan_compute(self.handle) != 0;
    }

    /// Keeps packed blobs as fp16.
    pub fn setUseFp16Packed(self: Option, enable: bool) void {
        c.ncnn_option_set_use_fp16_packed(self.handle, @intFromBool(enable));
    }

    pub fn useFp16Packed(self: Option) bool {
        return c.ncnn_option_get_use_fp16_packed(self.handle) != 0;
    }

    /// Computes in fp16 where the layer supports it.
    pub fn setUseFp16Arithmetic(self: Option, enable: bool) void {
        c.ncnn_option_set_use_fp16_arithmetic(self.handle, @intFromBool(enable));
    }

    pub fn useFp16Arithmetic(self: Option) bool {
        return c.ncnn_option_get_use_fp16_arithmetic(self.handle) != 0;
    }

    /// Keeps packed blobs as bf16.
    pub fn setUseBf16Packed(self: Option, enable: bool) void {
        c.ncnn_option_set_use_bf16_packed(self.handle, @intFromBool(enable));
    }

    pub fn useBf16Packed(self: Option) bool {
        return c.ncnn_option_get_use_bf16_packed(self.handle) != 0;
    }

    /// Keeps packed blobs as int8.
    pub fn setUseInt8Packed(self: Option, enable: bool) void {
        c.ncnn_option_set_use_int8_packed(self.handle, @intFromBool(enable));
    }

    pub fn useInt8Packed(self: Option) bool {
        return c.ncnn_option_get_use_int8_packed(self.handle) != 0;
    }

    /// Stores intermediate blobs as int8.
    pub fn setUseInt8Storage(self: Option, enable: bool) void {
        c.ncnn_option_set_use_int8_storage(self.handle, @intFromBool(enable));
    }

    pub fn useInt8Storage(self: Option) bool {
        return c.ncnn_option_get_use_int8_storage(self.handle) != 0;
    }

    /// Computes in int8 where the layer supports it.
    pub fn setUseInt8Arithmetic(self: Option, enable: bool) void {
        c.ncnn_option_set_use_int8_arithmetic(self.handle, @intFromBool(enable));
    }

    pub fn useInt8Arithmetic(self: Option) bool {
        return c.ncnn_option_get_use_int8_arithmetic(self.handle) != 0;
    }

    /// Keeps packed blobs as int16.
    pub fn setUseInt16Packed(self: Option, enable: bool) void {
        c.ncnn_option_set_use_int16_packed(self.handle, @intFromBool(enable));
    }

    pub fn useInt16Packed(self: Option) bool {
        return c.ncnn_option_get_use_int16_packed(self.handle) != 0;
    }

    /// Stores intermediate blobs as int16.
    pub fn setUseInt16Storage(self: Option, enable: bool) void {
        c.ncnn_option_set_use_int16_storage(self.handle, @intFromBool(enable));
    }

    pub fn useInt16Storage(self: Option) bool {
        return c.ncnn_option_get_use_int16_storage(self.handle) != 0;
    }
};
