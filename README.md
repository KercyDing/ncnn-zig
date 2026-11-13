# ncnn-zig

Zig wrapper and build integration for [ncnn](https://github.com/Tencent/ncnn).

The package:

- fetches a pinned ncnn source revision at build time;
- builds ncnn statically from source;
- exposes the generated C API as `ncnn.c`;
- exposes a higher-level Zig module as `ncnn`.

Requires Zig 0.17. ncnn is not vendored. Prebuilt archives are not shipped,
but `-Dprebuilt=<dir>` can consume one.

> [!WARNING]
> The wrapper is experimental. The build, smoke tests, and selected upstream
> tests pass, but not every ncnn API or configuration is covered.

## Use

Add the dependency:

```bash
zig fetch --save git+https://github.com/kercyding/ncnn-zig#v0.3.0
```

Link the static library and import the Zig module:

```zig
const ncnn_dep = b.dependency("ncnn_zig", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.linkLibrary(ncnn_dep.artifact("ncnn"));
exe.root_module.addImport("ncnn", ncnn_dep.module("ncnn"));
```

The Zig module covers the commonly used `Net`, `Extractor`, `Mat`, `Option`,
`ParamDict`, data-reader, allocator, pixel, and drawing APIs. The generated
raw C bindings remain available as `ncnn.c`.

### Prebuilt library

Point `prebuilt` at a directory containing `include/` and `lib/libncnn.a`
(`lib/ncnn.lib` on Windows):

```zig
const ncnn_dep = b.dependency("ncnn_zig", .{
    .target = target,
    .optimize = optimize,
    .prebuilt = .{ .cwd_relative = "/path/to/ncnn-prebuilt" },
});
```

This skips the upstream build and generates the bindings from the supplied
headers. The archive must match the target, Zig version, and ncnn options
(`layers`, ISA set, `int8`, `weight_quant`); otherwise the generated headers
and the linked library disagree.

## What can be customized

All options can be passed to `b.dependency(...)` or as `-D<name>`.

### Target and binary

Use Zig's normal `.target`, `.optimize`, and `.strip` options. ncnn is always
linked statically into the final executable.

### Compiled layers

`layers` controls which layer translation units are compiled:

```zig
.layers = "Input,Convolution,ReLU,Pooling,Softmax",
```

It accepts:

- `default`: upstream's default layer set;
- `all`: every layer known to ncnn-zig;
- a comma-separated allowlist.

The allowlist keeps ncnn's original layer IDs. Omitted layers therefore remain
compatible with existing `.param` files, but loading a model that needs an
omitted layer fails.

Static runtime and link dependencies are added automatically. For example,
`Convolution` may pull in `Gemm`. See [`tools/README.md`](tools/README.md) for
the layer tool details; use [`tools/layers.py`](tools/layers.py) to generate a
suggested list from one or more `.param` files:

```bash
python3 tools/layers.py model.param
python3 tools/layers.py models/*.param
```

The result is a build suggestion, not a replacement for testing the model.

### CPU instruction sets

On x86:

```zig
.runtime_x86 = true,
.x86_isa = "avx,avx2,avx512",
```

On AArch64:

```zig
.arm_isa = "asimdhp,asimddp,asimdfhm,bf16",
```

Use `all`, `off`, or a comma-separated subset. Selected variants are dispatched
at runtime. The base ncnn sources still use the target architecture baseline,
not `-march=native`.

### ncnn features

The following upstream features can be enabled or disabled:

```zig
.stdio = true,
.string = true,
.validation = true,
.threads = true,
.pixel = true,
.pixel_rotate = true,
.pixel_affine = true,
.pixel_drawing = true,
.batch = true,
.c_api = true,
.platform_api = true,
.int8 = true,
.weight_quant = true,
.bf16 = true,
```

Disable a feature only when the application does not use the corresponding
API or model path. For example, disabling `int8` removes quantized inference
support; disabling `pixel` removes pixel conversion support from the generated
C API. The Zig wrapper does not hide these configuration constraints.

Other build controls include:

- `force_inline`;
- `disable_rtti` and `disable_exception`;
- `openmp` and ncnn's built-in `simpleomp` runtime;
- `simpleocv`, `simplestl`, and `simplemath`;
- `benchmark` and `coverage`.

## What is not covered

This package does not provide:

- Vulkan support. `-Dvulkan=true` is rejected because the required shader
  toolchain is not part of the pinned upstream source package;
- prebuilt `.a`/`.lib` artifacts or shared-library builds; `-Dprebuilt` only
  consumes a directory you provide;
- model downloading, model conversion, preprocessing, postprocessing, or a
  model zoo;
- a complete high-level Zig abstraction over every ncnn API;
- automatic proof that a generated `layers` list is sufficient for every
  runtime code path;
- complete coverage of all ncnn options, architectures, and quantized paths.

Custom layers can still be implemented and registered by the application
through ncnn's runtime custom-layer API. They are not compiled by the
`layers` allowlist unless their implementation is supplied by the application.

Vulkan, unsupported upstream options, and model-specific behavior must be
handled by the application or by extending the build integration.

## Development

```bash
zig build               # build the static library
zig build bindings      # generate and type-check the wrapper
zig build test          # wrapper smoke tests
zig build test-upstream # selected upstream layer tests
zig build check-deps    # check layer dependencies against upstream sources
```

The upstream revision is pinned in `build.zig.zon`. ncnn C++ is compiled with
`sanitize_c = .off` because upstream uses type punning and pointer arithmetic
that UBSan rejects.

## License

The build integration and Zig wrapper are licensed under the
[MIT License](LICENSE).

ncnn is fetched from upstream and is licensed under the
[BSD 3-Clause License](https://github.com/Tencent/ncnn/blob/master/LICENSE.txt).
See [NOTICE](NOTICE).
