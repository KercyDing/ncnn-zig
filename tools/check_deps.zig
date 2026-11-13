//! Verifies `sources.runtime_dependencies` against the pinned upstream layer
//! sources. Run through `zig build check-deps`; the build step sets the cwd to
//! the ncnn checkout.

const std = @import("std");

const ncnn_sources = @import("ncnn_sources");

const Edge = struct {
    layer: []const u8,
    needs: []const u8,
};

const Layer = struct {
    lower: []const u8,
    name: []const u8,
};

pub fn main(init: std.process.Init) !void {
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = init.io;

    const layers = try arena.alloc(Layer, ncnn_sources.layers.len);
    for (ncnn_sources.layers, 0..) |layer, i| {
        layers[i] = .{
            .lower = try std.ascii.allocLowerString(arena, layer.name),
            .name = layer.name,
        };
    }

    var dir = try std.Io.Dir.cwd().openDir(io, "src/layer", .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(arena);
    defer walker.deinit();

    var found: std.ArrayList(Edge) = .empty;
    var errors: usize = 0;

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".cpp")) continue;
        if (!supportedPath(entry.path)) continue;

        const text = try entry.dir.readFileAlloc(io, entry.basename, arena, .unlimited);
        const owner: ?[]const u8 = if (layerForFile(layers, entry.basename)) |layer| layer.name else null;
        try scanCalls(arena, layers, text, owner, entry.path, &found, &errors);
    }

    errors += try compare(found.items);

    std.debug.print("check-deps: {d} runtime edges, {d} errors\n", .{ found.items.len, errors });
    if (errors != 0) std.process.exit(1);
}

/// Only the generic files and the x86/arm directories are part of this
/// package's layer table.
fn supportedPath(path: []const u8) bool {
    const parent = std.fs.path.dirname(path) orelse return true;
    return std.mem.eql(u8, parent, "x86") or std.mem.eql(u8, parent, "arm");
}

/// Maps a source file to its layer by the longest matching name prefix, e.g.
/// `convolutiondepthwise_x86.cpp` -> `ConvolutionDepthWise`.
fn layerForFile(layers: []const Layer, basename: []const u8) ?Layer {
    const stem = basename[0 .. basename.len - ".cpp".len];
    var best: ?Layer = null;
    for (layers) |layer| {
        if (stem.len < layer.lower.len) continue;
        if (!std.ascii.eqlIgnoreCase(stem[0..layer.lower.len], layer.lower)) continue;
        if (stem.len != layer.lower.len and stem[layer.lower.len] != '_') continue;
        if (best == null or layer.lower.len > best.?.lower.len) best = layer;
    }
    return best;
}

fn scanCalls(
    allocator: std.mem.Allocator,
    layers: []const Layer,
    text: []const u8,
    owner: ?[]const u8,
    path: []const u8,
    found: *std.ArrayList(Edge),
    errors: *usize,
) !void {
    var i: usize = 0;
    var line: usize = 1;

    while (i < text.len) {
        const c = text[i];
        if (c == '\n') {
            line += 1;
            i += 1;
            continue;
        }
        if (c == '/' and i + 1 < text.len) {
            if (text[i + 1] == '/') {
                i = skipLineComment(text, i);
                continue;
            }
            if (text[i + 1] == '*') {
                i = skipBlockComment(text, i, &line);
                continue;
            }
        }
        if (c == '"' or c == '\'') {
            i = skipQuoted(text, i, c, &line);
            continue;
        }
        if (!isIdentStart(c)) {
            i += 1;
            continue;
        }

        const begin = i;
        i += 1;
        while (i < text.len and isIdentContinue(text[i])) i += 1;
        const ident = text[begin..i];
        if (!std.mem.eql(u8, ident, "create_layer") and
            !std.mem.eql(u8, ident, "create_layer_cpu")) continue;

        const call = parseCall(text, i) orelse {
            std.debug.print("{s}:{d}: unsupported create_layer call\n", .{ path, line });
            errors.* += 1;
            continue;
        };
        i = call.end;

        const target = canonical(layers, call.name) orelse {
            std.debug.print("{s}:{d}: unknown layer target '{s}'\n", .{ path, line, call.name });
            errors.* += 1;
            continue;
        };
        const source = owner orelse {
            std.debug.print("{s}:{d}: create_layer outside a known layer file\n", .{ path, line });
            errors.* += 1;
            continue;
        };
        if (std.ascii.eqlIgnoreCase(source, target)) continue;
        try addEdge(allocator, found, source, target);
    }
}

fn parseCall(text: []const u8, start: usize) ?struct { end: usize, name: []const u8 } {
    var i = skipSpace(text, start);
    if (i >= text.len or text[i] != '(') return null;
    i = skipSpace(text, i + 1);

    if (i < text.len and text[i] == '"') {
        const begin = i + 1;
        i = begin;
        while (i < text.len and text[i] != '"') {
            if (text[i] == '\\') return null;
            i += 1;
        }
        if (i >= text.len) return null;
        const name = text[begin..i];
        i = skipSpace(text, i + 1);
        if (i >= text.len or text[i] != ')') return null;
        return .{ .end = i + 1, .name = name };
    }

    if (hasPrefix(text, i, "ncnn")) {
        if (i + 4 < text.len and isIdentContinue(text[i + 4])) return null;
        i = skipSpace(text, i + 4);
        if (!hasPrefix(text, i, "::")) return null;
        i = skipSpace(text, i + 2);
    }

    if (!hasPrefix(text, i, "LayerType")) return null;
    if (i + 9 < text.len and isIdentContinue(text[i + 9])) return null;
    i = skipSpace(text, i + 9);
    if (!hasPrefix(text, i, "::")) return null;
    i = skipSpace(text, i + 2);
    if (i >= text.len or !isIdentStart(text[i])) return null;

    const begin = i;
    i += 1;
    while (i < text.len and isIdentContinue(text[i])) i += 1;
    const name = text[begin..i];
    i = skipSpace(text, i);
    if (i >= text.len or text[i] != ')') return null;
    return .{ .end = i + 1, .name = name };
}

fn canonical(layers: []const Layer, name: []const u8) ?[]const u8 {
    for (layers) |layer| {
        if (std.mem.eql(u8, layer.name, name)) return layer.name;
    }
    return null;
}

fn addEdge(allocator: std.mem.Allocator, found: *std.ArrayList(Edge), layer: []const u8, needs: []const u8) !void {
    for (found.items) |edge| {
        if (std.mem.eql(u8, edge.layer, layer) and std.mem.eql(u8, edge.needs, needs)) return;
    }
    try found.append(allocator, .{ .layer = layer, .needs = needs });
}

fn compare(found: []const Edge) !usize {
    var missing: usize = 0;

    for (found) |edge| {
        var known = false;
        for (ncnn_sources.runtime_dependencies) |dep| {
            if (!std.mem.eql(u8, dep.layer, edge.layer)) continue;
            for (dep.needs) |need| {
                if (std.mem.eql(u8, need, edge.needs)) {
                    known = true;
                    break;
                }
            }
        }
        if (!known) {
            std.debug.print("missing runtime dependency: {s} -> {s}\n", .{ edge.layer, edge.needs });
            missing += 1;
        }
    }

    for (ncnn_sources.runtime_dependencies) |dep| {
        for (dep.needs) |need| {
            var present = false;
            for (found) |edge| {
                if (std.mem.eql(u8, edge.layer, dep.layer) and std.mem.eql(u8, edge.needs, need)) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                std.debug.print("warning: stale runtime dependency: {s} -> {s}\n", .{ dep.layer, need });
            }
        }
    }

    return missing;
}

fn hasPrefix(text: []const u8, i: usize, prefix: []const u8) bool {
    return i + prefix.len <= text.len and std.mem.eql(u8, text[i .. i + prefix.len], prefix);
}

fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_';
}

fn isIdentContinue(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn skipSpace(text: []const u8, start: usize) usize {
    var i = start;
    while (i < text.len and std.ascii.isWhitespace(text[i])) i += 1;
    return i;
}

fn skipLineComment(text: []const u8, start: usize) usize {
    var i = start + 2;
    while (i < text.len and text[i] != '\n') i += 1;
    return i;
}

fn skipBlockComment(text: []const u8, start: usize, line: *usize) usize {
    var i = start + 2;
    while (i < text.len) {
        if (text[i] == '\n') line.* += 1;
        if (text[i] == '*' and i + 1 < text.len and text[i + 1] == '/') return i + 2;
        i += 1;
    }
    return i;
}

fn skipQuoted(text: []const u8, start: usize, quote: u8, line: *usize) usize {
    var i = start + 1;
    while (i < text.len) {
        if (text[i] == '\n') line.* += 1;
        if (text[i] == '\\') {
            i += 2;
            continue;
        }
        if (text[i] == quote) return i + 1;
        i += 1;
    }
    return i;
}
