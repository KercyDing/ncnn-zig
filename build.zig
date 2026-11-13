const std = @import("std");
const builtin = @import("builtin");

const config = @import("src/config.zig");
const generate = @import("src/generate.zig");
const sources = @import("src/sources.zig");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "Strip debug symbols") orelse false;
    const options = config.Options.fromBuild(b);

    if (options.vulkan) {
        std.log.err("vulkan support is not implemented yet", .{});
        std.process.exit(1);
    }

    const upstream = b.dependency("ncnn", .{});
    const src_dir = upstream.path("src");
    const t = target.result;

    const arch = archDirName(t.cpu.arch);
    const is_x86 = if (arch) |a| std.mem.eql(u8, a, "x86") else false;
    const x86_dispatch = is_x86 and options.runtime_x86 and t.abi != .msvc;
    const x86_isa = x86: {
        if (!is_x86) break :x86 std.EnumSet(config.X86Isa).empty;
        if (!x86_dispatch) {
            if (!std.mem.eql(u8, options.x86_isa, "all") and !std.mem.eql(u8, options.x86_isa, "off"))
                fail("x86_isa requires runtime_x86=true and a non-MSVC ABI", .{});
            break :x86 std.EnumSet(config.X86Isa).empty;
        }
        const set = parseIsaOrExit(config.X86Isa, options.x86_isa, "x86_isa", x86_isa_names);
        validateX86Isa(set);
        break :x86 set;
    };
    const arm32 = switch (t.cpu.arch) {
        .arm, .armeb, .thumb, .thumbeb => true,
        else => false,
    };
    const arm64 = t.cpu.arch.isAarch64();
    if (arm64 and !options.runtime_x86 and !std.mem.eql(u8, options.arm_isa, "off"))
        fail("runtime_x86 is x86-only; pass -Darm_isa=off to disable aarch64 dispatch", .{});
    const arm_isa = if (arm64)
        parseIsaOrExit(config.ArmIsa, options.arm_isa, "arm_isa", arm_isa_names)
    else
        std.EnumSet(config.ArmIsa).empty;
    if (arm64) validateArmIsa(arm_isa, options.bf16);
    if (options.openmp and !options.threads)
        fail("openmp requires threads=true", .{});
    // simpleomp only needs pthread_once. Zig's mingw provides the primitive,
    // but upstream platform.h does not include pthread.h on Windows, so the
    // runtime source is patched for windows-gnu instead of pulling in
    // winpthreads.
    const windows_gnu = t.os.tag == .windows and t.abi == .gnu;
    const openmp = options.openmp and (t.os.tag != .windows or windows_gnu);
    // LLVM cannot emit SEH unwind info for the GNU asm blocks in the arm
    // kernels on aarch64-windows.
    const gnu_inline_asm = options.gnu_inline_asm and !(t.os.tag == .windows and t.cpu.arch.isAarch64());
    const layers = try resolveLayers(b, upstream, arch, options.layers);

    // ncnn's base sources must be built for the architecture baseline. Its
    // `#if __AVX__` style blocks belong to separately generated ISA variants,
    // so letting the host CPU features leak in (the default for a native
    // target) produces an inconsistent library.
    // 32-bit ARM needs NEON + VFPv4, which the default armv6 baseline lacks.
    const c_target = if (arch == null)
        target
    else if (arm32)
        armv7Target(b, target)
    else
        baselineTarget(b, target);
    const is_arm = arch != null and std.mem.eql(u8, arch.?, "arm");

    const mod = b.createModule(.{
        .target = c_target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
        .link_libcpp = true,
        // ncnn is third-party C++ with intentional type punning and pointer
        // arithmetic; upstream never builds it with UBSan instrumentation.
        .sanitize_c = .off,
    });

    const lib = b.addLibrary(.{
        .name = "ncnn",
        .root_module = mod,
    });

    // platform.h is generated from the upstream template so that new options
    // added by upstream are picked up without touching this repository.
    const platform = b.addConfigHeader(.{
        .style = .{ .cmake = src_dir.path(b, "platform.h.in") },
        .include_path = "platform.h",
    }, config.platformValues(
        options,
        sources.version.string,
        sources.version.number,
        x86_isa,
        is_arm,
        arm_isa,
        openmp,
        gnu_inline_asm,
    ));

    const generated = b.addWriteFiles();

    // x86 dispatch: upstream generates `<class>_x86_avx` and
    // `<class>_x86_avx512` from the baseline `_x86` sources by renaming the
    // class and include guard, then compiles them with the ISA flags.
    var avx_sources: std.ArrayList([]const u8) = .empty;
    defer avx_sources.deinit(b.allocator);
    var avx512_sources: std.ArrayList([]const u8) = .empty;
    defer avx512_sources.deinit(b.allocator);

    if (x86_isa.count() != 0) {
        for (layers) |*layer| {
            if (!layer.enabled) continue;
            const a = layer.arch orelse continue;
            if (!std.mem.eql(u8, a, "x86")) continue;

            const lower = try std.ascii.allocLowerString(b.allocator, layer.name);
            defer b.allocator.free(lower);

            const header_rel = try std.fmt.allocPrint(b.allocator, "src/layer/x86/{s}_x86.h", .{lower});
            _ = upstream.builder.root.statFile(b.graph.io, header_rel) catch continue;

            const base_h = try readUpstream(b, upstream, header_rel);
            defer b.allocator.free(base_h);
            const cpp_rel = try std.fmt.allocPrint(b.allocator, "src/layer/x86/{s}_x86.cpp", .{lower});
            const base_cpp = try readUpstream(b, upstream, cpp_rel);
            defer b.allocator.free(base_cpp);

            inline for (.{
                .{ .name = "avx", .tag = config.X86Isa.avx },
                .{ .name = "avx512", .tag = config.X86Isa.avx512 },
            }) |isa| {
                if (x86_isa.contains(isa.tag)) {
                    const header = try makeVariant(b.allocator, base_h, layer.name, lower, isa.name);
                    defer b.allocator.free(header);
                    const source = try makeVariant(b.allocator, base_cpp, layer.name, lower, isa.name);
                    defer b.allocator.free(source);

                    _ = generated.add(b.fmt("layer/x86/{s}_x86_{s}.h", .{ lower, isa.name }), header);
                    const source_path = b.fmt("layer/x86/{s}_x86_{s}.cpp", .{ lower, isa.name });
                    _ = generated.add(source_path, source);

                    if (comptime std.mem.eql(u8, isa.name, "avx")) {
                        layer.avx = true;
                        try avx_sources.append(b.allocator, source_path);
                    } else {
                        layer.avx512 = true;
                        try avx512_sources.append(b.allocator, source_path);
                    }
                }
            }
        }
    }

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
    if (x86_isa.contains(.avx) or x86_isa.contains(.avx512))
        mod.addIncludePath(generated.getDirectory().path(b, "layer/x86"));

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

    try flags.append(b.allocator, "-DNCNN_STATIC_DEFINE");

    // Expand `#pragma omp`; ncnn's simpleomp runtime provides the symbols.
    if (openmp) try flags.append(b.allocator, "-fopenmp");

    if (options.disable_rtti) try flags.append(b.allocator, "-fno-rtti");
    if (options.disable_exception) try flags.append(b.allocator, "-fno-exceptions");

    if (arch) |a| try appendArchFlags(b.allocator, &flags, a, t);

    var files: std.ArrayList([]const u8) = .empty;
    defer files.deinit(b.allocator);

    for (sources.core) |file| {
        // Windows builds use the patched copy added below.
        if (windows_gnu and openmp and std.mem.eql(u8, file, "simpleomp.cpp")) continue;
        try files.append(b.allocator, b.fmt("src/{s}", .{file}));
    }
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

    if (windows_gnu and openmp) {
        mod.addCSourceFiles(.{
            .root = b.path("cpp"),
            .files = &.{"simpleomp_win32.cpp"},
            .flags = flags.items,
        });
    }

    if (x86_isa.count() != 0) {
        if (avx_sources.items.len != 0) {
            const avx_flags = try flagsWithFeatures(b.allocator, flags.items, &avx_features);
            mod.addCSourceFiles(.{
                .root = generated.getDirectory(),
                .files = avx_sources.items,
                .flags = avx_flags,
            });
        }

        if (avx512_sources.items.len != 0) {
            const avx512_flags = try flagsWithFeatures(b.allocator, flags.items, &avx512_features);
            mod.addCSourceFiles(.{
                .root = generated.getDirectory(),
                .files = avx512_sources.items,
                .flags = avx512_flags,
            });
        }

        // Checked-in per-layer kernels that feed the base classes, e.g.
        // `convolution_x86_avx2.cpp`.
        const isa_sources = [_]struct {
            name: []const u8,
            features: []const []const u8,
            enabled: bool,
        }{
            .{ .name = "fma", .features = &fma_features, .enabled = x86_isa.contains(.fma) },
            .{ .name = "fma4", .features = &fma4_features, .enabled = x86_isa.contains(.fma4) },
            .{ .name = "xop", .features = &xop_features, .enabled = x86_isa.contains(.xop) },
            .{ .name = "avxvnni", .features = &avxvnni_features, .enabled = x86_isa.contains(.avxvnni) },
            .{ .name = "avxvnniint8", .features = &avxvnniint8_features, .enabled = x86_isa.contains(.avxvnniint8) },
            .{ .name = "avxneconvert", .features = &avxneconvert_features, .enabled = x86_isa.contains(.avxneconvert) },
            .{ .name = "f16c", .features = &f16c_features, .enabled = x86_isa.contains(.f16c) },
            .{ .name = "avx2", .features = &avx2_features, .enabled = x86_isa.contains(.avx2) },
            .{ .name = "avx512vnni", .features = &avx512vnni_features, .enabled = x86_isa.contains(.avx512vnni) },
            .{ .name = "avx512bf16", .features = &avx512bf16_features, .enabled = x86_isa.contains(.avx512bf16) },
            .{
                .name = "avx512vnni_avx512bf16",
                .features = &avx512vnni_avx512bf16_features,
                .enabled = x86_isa.contains(.avx512vnni) and x86_isa.contains(.avx512bf16),
            },
        };

        for (isa_sources) |isa| {
            if (!isa.enabled) continue;
            var group: std.ArrayList([]const u8) = .empty;
            defer group.deinit(b.allocator);

            for (layers) |layer| {
                if (!layer.enabled) continue;
                const a = layer.arch orelse continue;
                if (!std.mem.eql(u8, a, "x86")) continue;

                const lower = try std.ascii.allocLowerString(b.allocator, layer.name);
                defer b.allocator.free(lower);

                const source_rel = try std.fmt.allocPrint(
                    b.allocator,
                    "src/layer/x86/{s}_x86_{s}.cpp",
                    .{ lower, isa.name },
                );
                _ = upstream.builder.root.statFile(b.graph.io, source_rel) catch continue;
                try group.append(b.allocator, source_rel);
            }

            if (group.items.len == 0) continue;
            const group_flags = try flagsWithFeatures(b.allocator, flags.items, isa.features);
            mod.addCSourceFiles(.{
                .root = upstream.path(""),
                .files = group.items,
                .flags = group_flags,
            });
        }
    }

    if (is_arm) {
        var arm_isas: std.ArrayList(ArmIsa) = .empty;
        defer arm_isas.deinit(b.allocator);

        if (arm64) {
            // vfpv4 provides the fp16s fallback helpers and is always built;
            // the other ISAs are dispatched at runtime and user-selectable.
            try arm_isas.append(b.allocator, .{ .name = "vfpv4", .features = &arm_baseline_features });
            if (arm_isa.contains(.asimdhp))
                try arm_isas.append(b.allocator, .{ .name = "asimdhp", .features = &arm_fp16_features });
            if (arm_isa.contains(.asimddp))
                try arm_isas.append(b.allocator, .{ .name = "asimddp", .features = &arm_dotprod_features });
            if (arm_isa.contains(.asimdfhm))
                try arm_isas.append(b.allocator, .{ .name = "asimdfhm", .features = &arm_fp16fml_features });
            if (arm_isa.contains(.bf16))
                try arm_isas.append(b.allocator, .{ .name = "bf16", .features = &arm_bf16_features });
            if (arm_isa.contains(.i8mm))
                try arm_isas.append(b.allocator, .{ .name = "i8mm", .features = &arm_i8mm_features });
        } else {
            try arm_isas.append(b.allocator, .{ .name = "vfpv4", .features = &arm_vfpv4_features });
        }

        for (arm_isas.items) |isa| {
            var group: std.ArrayList([]const u8) = .empty;
            defer group.deinit(b.allocator);

            for (layers) |layer| {
                if (!layer.enabled) continue;
                const a = layer.arch orelse continue;
                if (!std.mem.eql(u8, a, "arm")) continue;

                const lower = try std.ascii.allocLowerString(b.allocator, layer.name);
                defer b.allocator.free(lower);

                const source_rel = try std.fmt.allocPrint(
                    b.allocator,
                    "src/layer/arm/{s}_arm_{s}.cpp",
                    .{ lower, isa.name },
                );
                _ = upstream.builder.root.statFile(b.graph.io, source_rel) catch continue;
                try group.append(b.allocator, source_rel);
            }

            if (group.items.len == 0) continue;
            const group_flags = try flagsWithFeatures(b.allocator, flags.items, isa.features);
            mod.addCSourceFiles(.{
                .root = upstream.path(""),
                .files = group.items,
                .flags = group_flags,
            });
        }
    }

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
    bindings.defineCMacro("NCNN_STATIC_DEFINE", null);
    bindings.linkLibrary(lib);

    const ncnn_c = b.addModule("ncnn_c", .{
        .root_source_file = bindings.output_file,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    ncnn_c.addImport("c_builtins", translate_c.module("c_builtins"));
    ncnn_c.addImport("helpers", translate_c.module("helpers"));
    ncnn_c.linkLibrary(lib);

    const ncnn = b.addModule("ncnn", .{
        .root_source_file = b.path("src/ncnn.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "ncnn_c", .module = ncnn_c },
        },
    });
    ncnn.linkLibrary(lib);

    const bindings_step = b.step("bindings", "Generate and type-check the Zig wrapper and its C bindings");
    bindings_step.dependOn(&bindings.run.step);
    bindings_step.dependOn(&b.addObject(.{
        .name = "ncnn_bindings",
        .root_module = ncnn,
    }).step);

    const test_step = b.step("test", "Run smoke tests");
    const test_mod = b.createModule(.{
        .root_source_file = b.path("test/smoke.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    test_mod.addImport("ncnn", ncnn);

    const tests = b.addTest(.{ .root_module = test_mod });
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);

    // Correct results alone also hold for a serial fallback, so check that
    // the runtime really dispatches the loop to multiple OS threads.
    if (openmp) {
        const simpleomp_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        simpleomp_mod.addCSourceFiles(.{
            .root = b.path("test"),
            .files = &.{"simpleomp.c"},
            .flags = &.{"-fopenmp"},
        });
        if (t.os.tag != .windows) simpleomp_mod.linkSystemLibrary("pthread", .{});
        simpleomp_mod.linkLibrary(lib);
        const simpleomp_test = b.addExecutable(.{
            .name = "test_simpleomp",
            .root_module = simpleomp_mod,
        });
        const run_simpleomp = b.addRunArtifact(simpleomp_test);
        test_step.dependOn(&run_simpleomp.step);
    }

    // Upstream's own layer tests compare the dispatched kernels against the
    // naive reference implementation, so they exercise the registry, the
    // generated headers and the dispatched kernels with real workloads.
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
    spec: []const u8,
) ![]generate.ResolvedLayer {
    const enabled = try resolveLayerSelection(b.allocator, spec);
    const list = try b.allocator.alloc(generate.ResolvedLayer, sources.layers.len);
    const io = b.graph.io;

    for (sources.layers, 0..) |layer, i| {
        list[i] = .{ .name = layer.name, .enabled = enabled[i] };
        if (!enabled[i]) continue;
        const a = arch orelse continue;

        const lower = try std.ascii.allocLowerString(b.allocator, layer.name);
        const rel = try std.fmt.allocPrint(b.allocator, "src/layer/{s}/{s}_{s}.cpp", .{ a, lower, a });
        _ = upstream.builder.root.statFile(io, rel) catch continue;
        list[i].arch = a;
    }

    return list;
}

/// Maps the `layers` option to one flag per table entry. Indices must keep
/// following the table order: they are baked into `layer_type_enum.h`.
fn resolveLayerSelection(allocator: std.mem.Allocator, spec: []const u8) ![]bool {
    const enabled = try allocator.alloc(bool, sources.layers.len);

    if (std.mem.eql(u8, spec, "default")) {
        for (sources.layers, enabled) |layer, *flag| flag.* = layer.enabled;
    } else if (std.mem.eql(u8, spec, "all")) {
        @memset(enabled, true);
    } else {
        @memset(enabled, false);
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |raw| {
            const name = std.mem.trim(u8, raw, " \t");
            if (name.len == 0) fail("layers: empty entry in '{s}'", .{spec});
            const index = findLayer(name) orelse fail("layers: unknown layer '{s}'", .{name});
            enabled[index] = true;
        }
    }

    // Pull in translation units that provide symbols referenced by enabled
    // layers; a static archive cannot resolve them on its own.
    var changed = true;
    while (changed) {
        changed = false;
        for (sources.dependencies) |dep| {
            const index = findLayer(dep.layer) orelse
                fail("layers: unknown dependency '{s}'", .{dep.layer});
            if (!enabled[index]) continue;
            for (dep.needs) |name| {
                const need = findLayer(name) orelse
                    fail("layers: unknown dependency '{s}'", .{name});
                if (!enabled[need]) {
                    enabled[need] = true;
                    changed = true;
                }
            }
        }
    }

    return enabled;
}

fn findLayer(name: []const u8) ?usize {
    for (sources.layers, 0..) |layer, i| {
        if (std.mem.eql(u8, layer.name, name)) return i;
    }
    return null;
}

/// Reads a file from the pinned upstream package.
fn readUpstream(
    b: *std.Build,
    upstream: *std.Build.Dependency,
    rel_path: []const u8,
) ![]u8 {
    // Paths are relative to the dependency's build root handle.
    const root = upstream.builder.root;
    const path = if (root.sub_path.len == 0)
        try b.allocator.dupe(u8, rel_path)
    else
        try std.fs.path.join(b.allocator, &.{ root.sub_path, rel_path });
    return root.root_dir.handle.readFileAlloc(b.graph.io, path, b.allocator, .unlimited);
}

/// Renames a baseline `_x86` source into an ISA variant, mirroring
/// `cmake/ncnn_generate_<isa>_source.cmake`.
fn makeVariant(
    allocator: std.mem.Allocator,
    data: []const u8,
    class: []const u8,
    lower: []const u8,
    isa: []const u8,
) ![]u8 {
    const upper = try std.ascii.allocUpperString(allocator, class);
    defer allocator.free(upper);
    const isa_upper = try std.ascii.allocUpperString(allocator, isa);
    defer allocator.free(isa_upper);

    const guard_from = try std.fmt.allocPrint(allocator, "LAYER_{s}_X86_H", .{upper});
    defer allocator.free(guard_from);
    const guard_to = try std.fmt.allocPrint(allocator, "LAYER_{s}_X86_{s}_H", .{ upper, isa_upper });
    defer allocator.free(guard_to);

    const class_from = try std.fmt.allocPrint(allocator, "{s}_x86", .{class});
    defer allocator.free(class_from);
    const class_to = try std.fmt.allocPrint(allocator, "{s}_x86_{s}", .{ class, isa });
    defer allocator.free(class_to);

    const include_from = try std.fmt.allocPrint(allocator, "#include \"{s}_x86.h\"", .{lower});
    defer allocator.free(include_from);
    const include_to = try std.fmt.allocPrint(allocator, "#include \"{s}_x86_{s}.h\"", .{ lower, isa });
    defer allocator.free(include_to);

    const step1 = try std.mem.replaceOwned(u8, allocator, data, guard_from, guard_to);
    defer allocator.free(step1);
    const step2 = try std.mem.replaceOwned(u8, allocator, step1, class_from, class_to);
    defer allocator.free(step2);
    return std.mem.replaceOwned(u8, allocator, step2, include_from, include_to);
}

/// Target features, lowest to highest. Zig passes an explicit `-target-feature`
/// set for the baseline CPU, which overrides `-mavx` style flags; pass the
/// variants as `-Xclang -target-feature` pairs so they win.
const sse4_features = [_][]const u8{ "ssse3", "sse4.1", "sse4.2" };
const avx_features = sse4_features ++ [_][]const u8{"avx"};
const fma_features = avx_features ++ [_][]const u8{ "fma", "f16c" };
const f16c_features = avx_features ++ [_][]const u8{"f16c"};
const avx2_features = fma_features ++ [_][]const u8{"avx2"};
const avx512_features = avx2_features ++ [_][]const u8{
    "avx512f", "avx512cd", "avx512bw", "avx512dq", "avx512vl",
};
const avx512vnni_features = avx512_features ++ [_][]const u8{"avx512vnni"};
const fma4_features = avx_features ++ [_][]const u8{ "xop", "fma4" };
const xop_features = avx_features ++ [_][]const u8{"xop"};
const avxvnni_features = avx2_features ++ [_][]const u8{"avxvnni"};
const avxvnniint8_features = avxvnni_features ++ [_][]const u8{"avxvnniint8"};
const avxneconvert_features = avx2_features ++ [_][]const u8{"avxneconvert"};
const avx512bf16_features = avx512_features ++ [_][]const u8{"avx512bf16"};
const avx512vnni_avx512bf16_features = avx512_features ++ [_][]const u8{ "avx512vnni", "avx512bf16" };

const ArmIsa = struct { name: []const u8, features: []const []const u8 };

const arm_baseline_features = [_][]const u8{};
const arm_vfpv4_features = [_][]const u8{ "neon", "vfp4" };
const arm_fp16_features = [_][]const u8{"fullfp16"};
const arm_dotprod_features = arm_fp16_features ++ [_][]const u8{"dotprod"};
const arm_fp16fml_features = arm_fp16_features ++ [_][]const u8{"fp16fml"};
const arm_bf16_features = arm_dotprod_features ++ [_][]const u8{"bf16"};
const arm_i8mm_features = arm_dotprod_features ++ [_][]const u8{"i8mm"};

/// Appends `features` to the common compile flags as cc1 target features.
fn flagsWithFeatures(
    allocator: std.mem.Allocator,
    base: []const []const u8,
    features: []const []const u8,
) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(allocator);
    try list.appendSlice(allocator, base);
    for (features) |feature| {
        const enable = try std.fmt.allocPrint(allocator, "+{s}", .{feature});
        try list.appendSlice(allocator, &.{ "-Xclang", "-target-feature", "-Xclang", enable });
    }
    return allocator.dupe([]const u8, list.items);
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
        if (t.cpu.arch == .x86) {
            try flags.appendSlice(allocator, &.{ "-Xclang", "-target-feature", "-Xclang", "+sse2" });
        }
        // Reciprocal approximations lose precision; upstream disables them too.
        if (t.abi != .msvc) try flags.append(allocator, "-mrecip=none");
    }
}

/// Resolves the same OS/ABI as `target` with an ARMv7 CPU that has NEON and
/// VFPv4, matching upstream's `-mfpu=neon-vfpv4`.
fn armv7Target(b: *std.Build, target: std.Build.ResolvedTarget) std.Build.ResolvedTarget {
    var query = target.query;
    query.cpu_model = .{ .explicit = &std.Target.arm.cpu.cortex_a7 };
    query.cpu_features_add = .empty;
    query.cpu_features_sub = .empty;
    return b.resolveTargetQuery(query);
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

const x86_isa_names = "avx, avx2, fma, fma4, xop, f16c, avxvnni, avxvnniint8, avxneconvert, avx512, avx512vnni, avx512bf16";
const arm_isa_names = "asimdhp, asimddp, asimdfhm, bf16, i8mm";

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.log.err(fmt, args);
    std.process.exit(1);
}

fn parseIsaOrExit(comptime E: type, spec: []const u8, label: []const u8, names: []const u8) std.EnumSet(E) {
    return config.parseIsaSet(E, spec) catch
        fail("{s} must be 'all', 'off', or a comma-separated subset of: {s}", .{ label, names });
}

fn validateX86Isa(set: std.EnumSet(config.X86Isa)) void {
    if ((set.contains(.fma) or set.contains(.fma4) or set.contains(.xop) or
        set.contains(.f16c) or set.contains(.avx2)) and !set.contains(.avx))
        fail("x86_isa: fma/fma4/xop/f16c/avx2 require avx", .{});
    if ((set.contains(.avxvnni) or set.contains(.avxvnniint8) or set.contains(.avxneconvert) or
        set.contains(.avx512)) and !set.contains(.avx2))
        fail("x86_isa: avxvnni/avxvnniint8/avxneconvert/avx512 require avx2", .{});
    if ((set.contains(.avx512vnni) or set.contains(.avx512bf16)) and !set.contains(.avx512))
        fail("x86_isa: avx512vnni/avx512bf16 require avx512", .{});
}

fn validateArmIsa(set: std.EnumSet(config.ArmIsa), bf16_feature: bool) void {
    if ((set.contains(.asimddp) or set.contains(.asimdfhm) or set.contains(.bf16) or set.contains(.i8mm)) and
        !set.contains(.asimdhp))
        fail("arm_isa: asimddp/asimdfhm/bf16/i8mm require asimdhp", .{});
    if ((set.contains(.bf16) or set.contains(.i8mm)) and
        !(set.contains(.asimddp) and set.contains(.asimdfhm)))
        fail("arm_isa: bf16/i8mm require asimddp and asimdfhm", .{});
    if (bf16_feature and !set.contains(.bf16))
        fail("arm_isa must include bf16 while bf16 is enabled; pass -Dbf16=false to drop bf16 support", .{});
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
