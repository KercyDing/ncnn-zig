const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// ncnn version compiled into the headers.
pub const header_string = c.NCNN_VERSION_STRING;
pub const header_number: i32 = c.NCNN_VERSION_NUMBER;

/// Returns the version of the linked ncnn runtime.
pub fn string() [:0]const u8 {
    return std.mem.span(c.ncnn_version());
}

/// Returns the numeric version of the linked ncnn runtime.
pub fn number() i32 {
    return c.ncnn_version_number();
}

/// Verifies that the headers used for the bindings match the linked runtime.
pub fn check() Error!void {
    if (number() != header_number) return Error.VersionMismatch;
}
