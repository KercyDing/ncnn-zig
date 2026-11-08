const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "Strip debug symbols") orelse false;
    // Only compile and link these x86 variants; ncnn picks between them at
    // runtime. Use "all", "off", or another comma-separated list with
    // -Dx86_isa=... to override.
    const x86_isa = b.option([]const u8, "x86_isa", "x86 dispatch subset: all, off, or comma-separated ISAs") orelse "avx,avx2,avx512";
    const arm_isa = b.option([]const u8, "arm_isa", "arm64 dispatch subset: all, off, or comma-separated ISAs") orelse "all";

    const dep = b.dependency("ncnn_zig", .{
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .x86_isa = x86_isa,
        .arm_isa = arm_isa,
    });

    const exe = b.addExecutable(.{
        .name = "relu",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = strip,
            .imports = &.{
                .{ .name = "ncnn", .module = dep.module("ncnn") },
            },
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    const step = b.step("run", "Run the example");
    step.dependOn(&run.step);
}
