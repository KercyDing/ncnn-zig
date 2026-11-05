//! Version helpers for the bindings and the linked library.

const std = @import("std");
const c = @import("ncnn_c");
const Error = @import("errors.zig").Error;

/// Version string baked into the bindings, e.g. "1.0.20261020".
pub const header_string = c.NCNN_VERSION_STRING;
/// Numeric version baked into the bindings, e.g. 20261020.
pub const header_number: i32 = c.NCNN_VERSION_NUMBER;

/// Returns the version string of the linked ncnn library.
pub fn string() [:0]const u8 {
    return std.mem.span(c.ncnn_version());
}

/// Returns the numeric version of the linked ncnn library.
pub fn number() i32 {
    return c.ncnn_version_number();
}

/// Fails when the linked library does not match the bindings.
pub fn check() Error!void {
    if (number() != header_number) return Error.VersionMismatch;
}
