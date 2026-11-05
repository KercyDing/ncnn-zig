const std = @import("std");
const builtin = @import("builtin");

const config = @import("src/config.zig");
const generate = @import("src/generate.zig");
const sources = @import("src/sources.zig");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const options = config.Options.fromBuild(b);

    if (options.vulkan) {
        std.log.err("vulkan support is not implemented yet", .{});
        std.process.exit(1);
    }
    if (options.runtime_cpu) {
        std.log.err("runtime CPU dispatch is not implemented yet", .{});
        std.process.exit(1);
    }

    const upstream = b.dependency("ncnn", .{});
    const src_dir = upstream.path("src");
    const t = target.result;

    const arch = archDirName(t.cpu.arch);
    const layers = try resolveLayers(b, upstream, arch);

    // ncnn's base sources must be built for the architecture baseline. Its
    // `#if __AVX__` style blocks belong to separately generated ISA variants,
    // so letting the host CPU features leak in (the default for a native
    // target) produces an inconsistent library.
    const c_target = if (arch != null) baselineTarget(b, target) else target;

    const mod = b.createModule(.{
        .target = c_target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
        // ncnn is third-party C++ with intentional type punning and pointer
        // arithmetic; upstream never builds it with UBSan instrumentation.
        .sanitize_c = .off,
    });

    const lib = b.addLibrary(.{
        .name = "ncnn",
        .linkage = options.linkage,
        .root_module = mod,
    });

    // platform.h is generated from the upstream template so that new options
    // added by upstream are picked up without touching this repository.
    const platform = b.addConfigHeader(.{
        .style = .{ .cmake = src_dir.path(b, "platform.h.in") },
        .include_path = "platform.h",
    }, config.platformValues(options, sources.version.string, sources.version.number));

    const generated = b.addWriteFiles();
    const export_header = generated.add("ncnn_export.h", config.export_header);
    const layer_declaration = generated.add("layer_declaration.h", try generate.layerDeclaration(b.allocator, layers));
    const layer_registry = generated.add("layer_registry.h", try generate.layerRegistry(b.allocator, layers));
    const layer_type_enum = generated.add("layer_type_enum.h", try generate.layerTypeEnum(b.allocator, layers));
    _ = generated.add("layer_shader_registry.h", generate.empty_shader_registry);
    _ = generated.add("layer_shader_type_enum.h", generate.empty_shader_type_enum);
    _ = generated.add("layer_shader_spv_data.h", generate.empty_shader_spv_data);

    mod.addIncludePath(platform.getOutputDir());
    mod.addIncludePath(generated.getDirectory());
    mod.addIncludePath(src_dir);
    if (arch) |a| mod.addIncludePath(src_dir.path(b, b.fmt("layer/{s}", .{a})));
    mod.addIncludePath(src_dir.path(b, "layer"));

    var flags: std.ArrayList([]const u8) = .empty;
    defer flags.deinit(b.allocator);

    try flags.appendSlice(b.allocator, &.{
        "-fno-strict-aliasing",
        "-Wall",
        "-Wextra",
        "-Wno-unused-function",
    });

    if (t.os.tag != .windows) {
        try flags.appendSlice(b.allocator, &.{
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
        });
    }

    if (t.os.tag == .windows) {
        try flags.appendSlice(b.allocator, &.{
            "-DNOMINMAX",
            "-D_SCL_SECURE_NO_WARNINGS",
            "-D_CRT_SECURE_NO_DEPRECATE",
        });
    }

    if (options.threads and t.os.tag != .windows) {
        try flags.append(b.allocator, "-pthread");
    }

    switch (options.linkage) {
        .static => try flags.append(b.allocator, "-DNCNN_STATIC_DEFINE"),
        .dynamic => try flags.append(b.allocator, "-DNCNN_BUILDING_LIBRARY"),
    }

    if (options.disable_rtti) try flags.append(b.allocator, "-fno-rtti");
    if (options.disable_exception) try flags.append(b.allocator, "-fno-exceptions");

    if (arch) |a| try appendArchFlags(b.allocator, &flags, a, t);

    var files: std.ArrayList([]const u8) = .empty;
    defer files.deinit(b.allocator);

    for (sources.core) |file| try files.append(b.allocator, b.fmt("src/{s}", .{file}));
    if (t.abi.isAndroid()) {
        for (sources.android) |file| try files.append(b.allocator, b.fmt("src/{s}", .{file}));
    }
    for (layers) |layer| {
        if (!layer.enabled) continue;
        const lower = try std.ascii.allocLowerString(b.allocator, layer.name);
        try files.append(b.allocator, b.fmt("src/layer/{s}.cpp", .{lower}));
        if (layer.arch) |a| {
            try files.append(b.allocator, b.fmt("src/layer/{s}/{s}_{s}.cpp", .{ a, lower, a }));
        }
    }

    mod.addCSourceFiles(.{
        .root = upstream.path(""),
        .files = files.items,
        .flags = flags.items,
    });

    if (options.threads and t.os.tag != .windows) {
        lib.root_module.linkSystemLibrary("pthread", .{});
    }

    lib.installHeadersDirectory(src_dir, "", .{ .include_extensions = &.{".h"} });
    lib.installHeader(platform.getOutputFile(), "platform.h");
    lib.installHeader(export_header, "ncnn_export.h");
    lib.installHeader(layer_declaration, "layer_declaration.h");
    lib.installHeader(layer_registry, "layer_registry.h");
    lib.installHeader(layer_type_enum, "layer_type_enum.h");

    b.installArtifact(lib);

    // Zig bindings for the C API, generated from the upstream header with
    // translate-c instead of `@cImport`.
    const Translator = @import("translate_c").Translator;
    const translate_c = b.dependency("translate_c", .{});
    const bindings: Translator = .init(translate_c, .{
        .name = "ncnn",
        .c_source_file = src_dir.path(b, "c_api.h"),
        .target = target,
        .optimize = optimize,
        // Fail the build if upstream headers start producing translation
        // warnings, instead of silently emitting degraded bindings.
        .warnings = .@"error",
    });
    // mingw headers use GCC-specific pragmas that aro does not know; the
    // resulting meta-warning must not fail an otherwise clean translation.
    bindings.run.addArg("-Wno-unknown-warning-option");
    bindings.addConfigHeader(platform);
    bindings.addIncludePath(generated.getDirectory());
    bindings.addIncludePath(src_dir);
    if (options.linkage == .static) bindings.defineCMacro("NCNN_STATIC_DEFINE", null);
    bindings.linkLibrary(lib);

    const ncnn_module = b.addModule("ncnn", .{
        .root_source_file = bindings.output_file,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ncnn_module.addImport("c_builtins", translate_c.module("c_builtins"));
    ncnn_module.addImport("helpers", translate_c.module("helpers"));
    ncnn_module.linkLibrary(lib);

    const bindings_step = b.step("bindings", "Generate and type-check Zig bindings for the ncnn C API");
    bindings_step.dependOn(&bindings.run.step);
    bindings_step.dependOn(&b.addObject(.{
        .name = "ncnn_bindings",
        .root_module = ncnn_module,
    }).step);

    const test_step = b.step("test", "Run smoke tests");
    const test_mod = b.createModule(.{
        .root_source_file = b.path("test/smoke.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    test_mod.addImport("ncnn", ncnn_module);

    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);

    // Upstream's own layer tests compare the dispatched kernels against the
    // naive reference implementation, so they exercise the registry, the
    // generated headers and the generic kernels with real workloads.
    const upstream_test_step = b.step("test-upstream", "Run upstream ncnn layer tests");
    const upstream_test_names = [_][]const u8{
        "test_absval",
        "test_relu",
        "test_pooling",
        "test_convolution",
        "test_innerproduct",
        "test_softmax",
    };
    for (upstream_test_names) |name| {
        const upstream_test_mod = b.createModule(.{
            .target = c_target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
            .sanitize_c = .off,
        });
        upstream_test_mod.addIncludePath(upstream.path("tests"));
        upstream_test_mod.addCSourceFiles(.{
            .root = upstream.path(""),
            .files = &.{ "tests/testutil.cpp", b.fmt("tests/{s}.cpp", .{name}) },
            .flags = flags.items,
        });
        upstream_test_mod.linkLibrary(lib);

        const upstream_test_exe = b.addExecutable(.{
            .name = name,
            .root_module = upstream_test_mod,
        });
        const run_upstream_test = b.addRunArtifact(upstream_test_exe);
        upstream_test_step.dependOn(&run_upstream_test.step);
    }
}

/// Maps a Zig CPU architecture to ncnn's `src/layer/<arch>` directory.
fn archDirName(cpu: std.Target.Cpu.Arch) ?[]const u8 {
    return switch (cpu) {
        .x86, .x86_64 => "x86",
        .arm, .armeb, .thumb, .thumbeb, .aarch64, .aarch64_be => "arm",
        else => null,
    };
}

/// Discovers `src/layer/<arch>/<layer>_<arch>.cpp` in the pinned upstream
/// tree, so bumping the dependency picks up new kernels automatically.
fn resolveLayers(
    b: *std.Build,
    upstream: *std.Build.Dependency,
    arch: ?[]const u8,
) ![]generate.ResolvedLayer {
    const list = try b.allocator.alloc(generate.ResolvedLayer, sources.layers.len);
    const io = b.graph.io;

    for (sources.layers, 0..) |layer, i| {
        list[i] = .{ .name = layer.name, .enabled = layer.enabled };
        if (!layer.enabled) continue;
        const a = arch orelse continue;

        const lower = try std.ascii.allocLowerString(b.allocator, layer.name);
        const rel = try std.fmt.allocPrint(b.allocator, "src/layer/{s}/{s}_{s}.cpp", .{ a, lower, a });
        _ = upstream.builder.root.statFile(io, rel) catch continue;
        list[i].arch = a;
    }

    return list;
}

fn appendArchFlags(
    allocator: std.mem.Allocator,
    flags: *std.ArrayList([]const u8),
    arch: []const u8,
    t: std.Target,
) !void {
    if (std.mem.eql(u8, arch, "x86")) {
        // 32-bit x86 baseline is i386; ncnn requires SSE2 for its `_x86`
        // kernels. x86_64 already includes it.
        try flags.appendSlice(allocator, &.{ "-msse2", "-msse" });
        // Reciprocal approximations lose precision; upstream disables them too.
        if (t.abi != .msvc) try flags.append(allocator, "-mrecip=none");
    } else if (std.mem.eql(u8, arch, "arm") and (t.cpu.arch == .arm or t.cpu.arch == .armeb or t.cpu.arch == .thumb or t.cpu.arch == .thumbeb)) {
        try flags.append(allocator, "-mfpu=neon-vfpv4");
    }
}

/// Resolves the same OS/ABI as `target` with the architecture baseline CPU
/// model, matching how upstream builds its base sources.
fn baselineTarget(b: *std.Build, target: std.Build.ResolvedTarget) std.Build.ResolvedTarget {
    var query = target.query;
    query.cpu_model = .baseline;
    query.cpu_features_add = .empty;
    query.cpu_features_sub = .empty;
    return b.resolveTargetQuery(query);
}

comptime {
    const minimum = std.SemanticVersion.parse("0.17.0") catch unreachable;

    if (builtin.zig_version.order(minimum) == .lt) {
        @compileError(
            \\Your version of Zig is too old.
            \\Minimum required version: 0.17.0
        );
    }
}
