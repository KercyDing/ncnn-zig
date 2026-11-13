//! Build options and the values substituted into upstream `src/platform.h.in`.

const std = @import("std");

/// x86 runtime-dispatch ISAs that have upstream variant sources.
pub const X86Isa = enum {
    avx,
    avx2,
    fma,
    fma4,
    xop,
    f16c,
    avxvnni,
    avxvnniint8,
    avxneconvert,
    avx512,
    avx512vnni,
    avx512bf16,
};

/// aarch64 runtime-dispatch ISAs that have upstream variant sources.
/// The vfpv4 baseline kernels are always built and are not selectable.
pub const ArmIsa = enum {
    asimdhp,
    asimddp,
    asimdfhm,
    bf16,
    i8mm,
};

pub fn parseIsaSet(comptime E: type, spec: []const u8) error{InvalidIsa}!std.EnumSet(E) {
    const trimmed = std.mem.trim(u8, spec, " \t");
    if (std.mem.eql(u8, trimmed, "all")) return std.EnumSet(E).full;
    if (trimmed.len == 0 or std.mem.eql(u8, trimmed, "off")) return std.EnumSet(E).empty;

    var set: std.EnumSet(E) = .empty;
    var it = std.mem.splitScalar(u8, trimmed, ',');
    while (it.next()) |raw| {
        const name = std.mem.trim(u8, raw, " \t");
        if (name.len == 0) return error.InvalidIsa;
        const tag = std.meta.stringToEnum(E, name) orelse return error.InvalidIsa;
        set.insert(tag);
    }
    return set;
}

pub const Options = struct {
    // Feature switches mirroring upstream CMake options of the same name.
    stdio: bool = true,
    string: bool = true,
    validation: bool = true,
    threads: bool = true,
    pixel: bool = true,
    pixel_rotate: bool = true,
    pixel_affine: bool = true,
    pixel_drawing: bool = true,
    batch: bool = true,
    c_api: bool = true,
    platform_api: bool = true,
    int8: bool = true,
    weight_quant: bool = true,
    bf16: bool = true,
    force_inline: bool = true,

    simpleocv: bool = false,
    /// Parallel loops via ncnn's built-in simpleomp runtime.
    openmp: bool = true,
    simplestl: bool = false,
    simplemath: bool = false,
    winxp: bool = false,

    benchmark: bool = false,
    coverage: bool = false,

    vulkan: bool = false,
    simplevk: bool = true,
    system_glslang: bool = false,

    /// Runtime CPU dispatch for x86 (AVX/AVX2/AVX512 variants). aarch64
    /// dispatch is controlled by `arm_isa`.
    runtime_x86: bool = true,

    /// x86 dispatch subset: `all`, `off`, or comma-separated `X86Isa` names.
    x86_isa: []const u8 = "all",
    /// aarch64 dispatch subset: `all`, `off`, or comma-separated `ArmIsa` names.
    arm_isa: []const u8 = "all",

    /// Layer allowlist: `default` (upstream defaults), `all`, or a
    /// comma-separated list of layer class names.
    layers: []const u8 = "default",

    gnu_inline_asm: bool = true,

    disable_rtti: bool = false,
    disable_exception: bool = false,

    pub fn fromBuild(b: *std.Build) Options {
        var o: Options = .{};
        inline for (.{
            "stdio",        "string",            "validation", "threads",      "pixel",  "pixel_rotate", "pixel_affine",   "pixel_drawing", "batch",
            "c_api",        "platform_api",      "int8",       "weight_quant", "bf16",   "force_inline", "simpleocv",      "openmp",        "simplestl",
            "simplemath",   "winxp",             "benchmark",  "coverage",     "vulkan", "simplevk",     "system_glslang", "runtime_x86",   "gnu_inline_asm",
            "disable_rtti", "disable_exception",
        }) |name| {
            @field(o, name) = b.option(bool, name, "ncnn " ++ name) orelse @field(o, name);
        }
        o.x86_isa = b.option([]const u8, "x86_isa", "x86 dispatch subset: all, off, or comma-separated ISAs") orelse o.x86_isa;
        o.arm_isa = b.option([]const u8, "arm_isa", "arm64 dispatch subset: all, off, or comma-separated ISAs") orelse o.arm_isa;
        o.layers = b.option([]const u8, "layers", "layer allowlist: default, all, or comma-separated names") orelse o.layers;
        return o;
    }
};

pub const PlatformValues = struct {
    NCNN_STDIO: bool,
    NCNN_STRING: bool,
    NCNN_VALIDATION: bool,
    NCNN_SIMPLEOCV: bool,
    NCNN_SIMPLEOMP: bool,
    NCNN_SIMPLESTL: bool,
    NCNN_SIMPLEMATH: bool,
    NCNN_THREADS: bool,
    NCNN_BENCHMARK: bool,
    NCNN_C_API: bool,
    NCNN_PLATFORM_API: bool,
    NCNN_BATCH: bool,
    NCNN_WINXP: bool,
    NCNN_PIXEL: bool,
    NCNN_PIXEL_ROTATE: bool,
    NCNN_PIXEL_AFFINE: bool,
    NCNN_PIXEL_DRAWING: bool,
    NCNN_VULKAN: bool,
    NCNN_COVERAGE: bool,
    NCNN_SIMPLEVK: bool,
    NCNN_SYSTEM_GLSLANG: bool,
    NCNN_RUNTIME_CPU: bool,
    NCNN_GNU_INLINE_ASM: bool,
    NCNN_AVX: bool,
    NCNN_XOP: bool,
    NCNN_FMA: bool,
    NCNN_FMA4: bool,
    NCNN_F16C: bool,
    NCNN_AVX2: bool,
    NCNN_AVXVNNI: bool,
    NCNN_AVXVNNIINT8: bool,
    NCNN_AVXVNNIINT16: bool,
    NCNN_AVXNECONVERT: bool,
    NCNN_AVX512: bool,
    NCNN_AVX512VNNI: bool,
    NCNN_AVX512BF16: bool,
    NCNN_AVX512FP16: bool,
    NCNN_VFPV4: bool,
    NCNN_ARM82: bool,
    NCNN_ARM82DOT: bool,
    NCNN_ARM82FP16FML: bool,
    NCNN_ARM84BF16: bool,
    NCNN_ARM84I8MM: bool,
    NCNN_ARM86SVE: bool,
    NCNN_ARM86SVE2: bool,
    NCNN_ARM86SVEBF16: bool,
    NCNN_ARM86SVEI8MM: bool,
    NCNN_ARM86SVEF32MM: bool,
    NCNN_MSA: bool,
    NCNN_LSX: bool,
    NCNN_LASX: bool,
    NCNN_MMI: bool,
    NCNN_RVV: bool,
    NCNN_ZFH: bool,
    NCNN_ZVFH: bool,
    NCNN_ZVFBFMIN: bool,
    NCNN_ZVFBFWMA: bool,
    NCNN_XTHEADVECTOR: bool,
    NCNN_INT8: bool,
    NCNN_WEIGHT_QUANT: bool,
    NCNN_BF16: bool,
    NCNN_FORCE_INLINE: bool,
    NCNN_VERSION_STRING: []const u8,
    NCNN_VERSION_NUMBER: i64,
};

pub fn platformValues(
    o: Options,
    version_string: []const u8,
    version_number: i64,
    x86_isa: std.EnumSet(X86Isa),
    is_arm: bool,
    arm_isa: std.EnumSet(ArmIsa),
    openmp: bool,
    gnu_inline_asm: bool,
) PlatformValues {
    return .{
        .NCNN_STDIO = o.stdio,
        .NCNN_STRING = o.string,
        .NCNN_VALIDATION = o.validation,
        .NCNN_SIMPLEOCV = o.simpleocv,
        .NCNN_SIMPLEOMP = openmp,
        .NCNN_SIMPLESTL = o.simplestl,
        .NCNN_SIMPLEMATH = o.simplemath,
        .NCNN_THREADS = o.threads,
        .NCNN_BENCHMARK = o.benchmark,
        .NCNN_C_API = o.c_api,
        .NCNN_PLATFORM_API = o.platform_api,
        .NCNN_BATCH = o.batch,
        .NCNN_WINXP = o.winxp,
        .NCNN_PIXEL = o.pixel,
        .NCNN_PIXEL_ROTATE = o.pixel_rotate,
        .NCNN_PIXEL_AFFINE = o.pixel_affine,
        .NCNN_PIXEL_DRAWING = o.pixel_drawing,
        .NCNN_VULKAN = o.vulkan,
        .NCNN_COVERAGE = o.coverage,
        .NCNN_SIMPLEVK = o.simplevk,
        .NCNN_SYSTEM_GLSLANG = o.system_glslang,
        .NCNN_RUNTIME_CPU = x86_isa.count() != 0 or arm_isa.count() != 0,
        .NCNN_GNU_INLINE_ASM = gnu_inline_asm,
        // x86 runtime dispatch: base code calls the dispatched variants.
        // AVX512-FP16 and AVXVNNI-INT16 have no upstream sources and stay off.
        .NCNN_AVX = x86_isa.contains(.avx),
        .NCNN_XOP = x86_isa.contains(.xop),
        .NCNN_FMA = x86_isa.contains(.fma),
        .NCNN_FMA4 = x86_isa.contains(.fma4),
        .NCNN_F16C = x86_isa.contains(.f16c),
        .NCNN_AVX2 = x86_isa.contains(.avx2),
        .NCNN_AVXVNNI = x86_isa.contains(.avxvnni),
        .NCNN_AVXVNNIINT8 = x86_isa.contains(.avxvnniint8),
        .NCNN_AVXVNNIINT16 = false,
        .NCNN_AVXNECONVERT = x86_isa.contains(.avxneconvert),
        .NCNN_AVX512 = x86_isa.contains(.avx512),
        .NCNN_AVX512VNNI = x86_isa.contains(.avx512vnni),
        .NCNN_AVX512BF16 = x86_isa.contains(.avx512bf16),
        .NCNN_AVX512FP16 = false,
        .NCNN_VFPV4 = is_arm,
        .NCNN_ARM82 = arm_isa.contains(.asimdhp),
        .NCNN_ARM82DOT = arm_isa.contains(.asimddp),
        .NCNN_ARM82FP16FML = arm_isa.contains(.asimdfhm),
        .NCNN_ARM84BF16 = arm_isa.contains(.bf16),
        .NCNN_ARM84I8MM = arm_isa.contains(.i8mm),
        .NCNN_ARM86SVE = false,
        .NCNN_ARM86SVE2 = false,
        .NCNN_ARM86SVEBF16 = false,
        .NCNN_ARM86SVEI8MM = false,
        .NCNN_ARM86SVEF32MM = false,
        .NCNN_MSA = false,
        .NCNN_LSX = false,
        .NCNN_LASX = false,
        .NCNN_MMI = false,
        .NCNN_RVV = false,
        .NCNN_ZFH = false,
        .NCNN_ZVFH = false,
        .NCNN_ZVFBFMIN = false,
        .NCNN_ZVFBFWMA = false,
        .NCNN_XTHEADVECTOR = false,
        .NCNN_INT8 = o.int8,
        .NCNN_WEIGHT_QUANT = o.weight_quant,
        .NCNN_BF16 = o.bf16,
        .NCNN_FORCE_INLINE = o.force_inline,
        .NCNN_VERSION_STRING = version_string,
        .NCNN_VERSION_NUMBER = version_number,
    };
}

/// `ncnn_export.h`, normally emitted by CMake's `GenerateExportHeader`.
pub const export_header =
    \\#ifndef NCNN_EXPORT_H
    \\#define NCNN_EXPORT_H
    \\
    \\#ifdef NCNN_STATIC_DEFINE
    \\#define NCNN_EXPORT
    \\#define NCNN_NO_EXPORT
    \\#elif defined(_WIN32) || defined(__CYGWIN__)
    \\#ifdef NCNN_BUILDING_LIBRARY
    \\#define NCNN_EXPORT __declspec(dllexport)
    \\#else
    \\#define NCNN_EXPORT __declspec(dllimport)
    \\#endif
    \\#define NCNN_NO_EXPORT
    \\#else
    \\#define NCNN_EXPORT __attribute__((visibility("default")))
    \\#define NCNN_NO_EXPORT __attribute__((visibility("hidden")))
    \\#endif
    \\
    \\#endif
    \\
;
