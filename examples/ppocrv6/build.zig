const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const strip = b.option(bool, "strip", "Strip debug symbols") orelse false;
    const x86_isa = b.option([]const u8, "x86_isa", "x86 dispatch subset: all, off, or comma-separated ISAs") orelse "avx,avx2,avx512";
    const arm_isa = b.option([]const u8, "arm_isa", "arm64 dispatch subset: all, off, or comma-separated ISAs") orelse "all";

    const ncnn = b.dependency("ncnn", .{
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .x86_isa = x86_isa,
        .arm_isa = arm_isa,
    });
    const zignal = b.dependency("zignal", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "ppocrv6",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = strip,
            .imports = &.{
                .{ .name = "ncnn", .module = ncnn.module("ncnn") },
                .{ .name = "zignal", .module = zignal.module("zignal") },
            },
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();
    const step = b.step("run", "Run PP-OCRv6 text-line recognition");
    step.dependOn(&run.step);
}
