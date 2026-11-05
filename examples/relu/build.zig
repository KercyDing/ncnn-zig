const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const dep = b.dependency("ncnn", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "relu",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ncnn", .module = dep.module("ncnn") },
            },
        }),
    });
    b.installArtifact(exe);

    const run = b.addRunArtifact(exe);
    const step = b.step("run", "Run the example");
    step.dependOn(&run.step);
}
