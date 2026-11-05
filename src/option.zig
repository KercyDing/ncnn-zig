const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// Owns an `ncnn::Option`. Changes apply when passed to `Net.setOption` or
/// `Extractor.setOption`.
pub const Option = struct {
    handle: c.ncnn_option_t,

    pub fn init() Error!Option {
        return .{ .handle = c.ncnn_option_create() orelse return Error.CreateOptionFailed };
    }

    pub fn deinit(self: *Option) void {
        if (self.handle) |handle| c.ncnn_option_destroy(handle);
        self.handle = null;
    }

    pub fn raw(self: Option) c.ncnn_option_t {
        return self.handle;
    }

    pub fn setNumThreads(self: Option, num_threads: i32) void {
        c.ncnn_option_set_num_threads(self.handle, num_threads);
    }

    pub fn numThreads(self: Option) i32 {
        return c.ncnn_option_get_num_threads(self.handle);
    }

    pub fn setUsePackingLayout(self: Option, enable: bool) void {
        c.ncnn_option_set_use_packing_layout(self.handle, @intFromBool(enable));
    }

    pub fn usePackingLayout(self: Option) bool {
        return c.ncnn_option_get_use_packing_layout(self.handle) != 0;
    }

    pub fn setUseWinogradConvolution(self: Option, enable: bool) void {
        c.ncnn_option_set_use_winograd_convolution(self.handle, @intFromBool(enable));
    }

    pub fn useWinogradConvolution(self: Option) bool {
        return c.ncnn_option_get_use_winograd_convolution(self.handle) != 0;
    }

    pub fn setUseSgemmConvolution(self: Option, enable: bool) void {
        c.ncnn_option_set_use_sgemm_convolution(self.handle, @intFromBool(enable));
    }

    pub fn useSgemmConvolution(self: Option) bool {
        return c.ncnn_option_get_use_sgemm_convolution(self.handle) != 0;
    }

    pub fn setUseFp16Storage(self: Option, enable: bool) void {
        c.ncnn_option_set_use_fp16_storage(self.handle, @intFromBool(enable));
    }

    pub fn useFp16Storage(self: Option) bool {
        return c.ncnn_option_get_use_fp16_storage(self.handle) != 0;
    }

    pub fn setUseBf16Storage(self: Option, enable: bool) void {
        c.ncnn_option_set_use_bf16_storage(self.handle, @intFromBool(enable));
    }

    pub fn useBf16Storage(self: Option) bool {
        return c.ncnn_option_get_use_bf16_storage(self.handle) != 0;
    }

    pub fn setUseLocalPoolAllocator(self: Option, enable: bool) void {
        c.ncnn_option_set_use_local_pool_allocator(self.handle, @intFromBool(enable));
    }

    pub fn useLocalPoolAllocator(self: Option) bool {
        return c.ncnn_option_get_use_local_pool_allocator(self.handle) != 0;
    }

    pub fn setUseVulkanCompute(self: Option, enable: bool) void {
        c.ncnn_option_set_use_vulkan_compute(self.handle, @intFromBool(enable));
    }

    pub fn useVulkanCompute(self: Option) bool {
        return c.ncnn_option_get_use_vulkan_compute(self.handle) != 0;
    }
};
