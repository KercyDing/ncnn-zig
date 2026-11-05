# ncnn.zig

Zig wrapper for [ncnn](https://github.com/Tencent/ncnn)'s standalone
[C API](https://github.com/Tencent/ncnn/blob/master/src/c_api.h).

- Builds from source. No CMake, no prebuilt binaries.
- The upstream revision is pinned in `build.zig.zon`.
- Upstream sources are not vendored here.
- Requires Zig 0.17.

> [!WARNING]
> The Zig wrapper is AI-assisted and only partially verified. The shipped
> build and tests pass, but most wrapper functions, options and quantized
> paths have not been exercised. Treat it as experimental.

## Usage

Add the package:

```bash
zig fetch --save=ncnn git+https://github.com/kercyding/ncnn.zig
```

Link it from `build.zig`:

```zig
const ncnn_dep = b.dependency("ncnn", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.linkLibrary(ncnn_dep.artifact("ncnn"));
```

The package also exports a Zig module named `ncnn`:

```zig
exe.root_module.addImport("ncnn", ncnn_dep.module("ncnn"));
```

The module wraps `Net`, `Extractor`, `Mat`, `Option`, `ParamDict`, custom
layers, data readers and allocators, plus pixel conversion and drawing
helpers. It links the library for you.

```zig
const ncnn = @import("ncnn");

var net = try ncnn.Net.init();
defer net.deinit();
try net.loadParamFile("model.param");
try net.loadModelFile("model.bin");

var extractor = try net.createExtractor();
defer extractor.deinit();

var input = try ncnn.Mat.fromSlice3d(224, 224, 3, pixels);
defer input.deinit();
try extractor.input("data", input);

var output = try extractor.extract("prob");
defer output.deinit();
```

The raw C API, generated with
[translate-c](https://codeberg.org/ziglang/translate-c), stays available as
`ncnn.c`.

Headers (`platform.h`, `c_api.h`, `layer/…`) are installed with the library.

## Build options

Options mirror the upstream CMake switches. Pass them to `b.dependency(...)` or
as `-D<name>`.

| Option | Default | |
| --- | --- | --- |
| `linkage` | `static` | `static` or `dynamic` |
| `stdio`, `string`, `validation` | `true` | file loading, type names, param validation |
| `threads` | `true` | thread pool |
| `pixel`, `pixel_rotate`, `pixel_affine`, `pixel_drawing` | `true` | image helpers |
| `batch` | `true` | batch inference |
| `c_api`, `platform_api` | `true` | C API, platform helpers |
| `int8`, `weight_quant`, `bf16` | `true` | quantized paths |
| `force_inline` | `true` | force-inline hot helpers |
| `benchmark`, `coverage` | `false` | upstream debug switches |
| `simpleocv`, `simpleomp`, `simplestl`, `simplemath` | `false` | minimal stubs |
| `winxp` | `false` | Windows XP compatibility |
| `disable_rtti`, `disable_exception` | `false` | build without RTTI/exceptions |
| `runtime_cpu` | `true` | x86 AVX/AVX512 dispatch |

```zig
const ncnn_dep = b.dependency("ncnn", .{
    .target = target,
    .optimize = optimize,
    .linkage = .dynamic,
    .pixel = false,
});
```

## Status

- Built for the architecture baseline: `x86_64`/`i386` + SSE2, `aarch64` NEON.
- This matches how upstream compiles its base sources.
- Optimized `src/layer/x86` and `src/layer/arm` kernels are included when they
  exist.
- x86 runtime dispatch is built: AVX, FMA, F16C, AVX2, AVX512, AVX512VNNI,
  AVX512BF16, FMA4, XOP, AVX-VNNI, AVX-VNNI-INT8 and AVX-NE-CONVERT variants
  are generated from the upstream sources and picked by ncnn's CPU check.
- ARM: 32-bit builds target ARMv7 NEON/VFPv4; AArch64 also compiles FP16,
  dotprod, fp16fml, bf16 and i8mm variants for runtime dispatch.
- Not built: AVX512-FP16, AVX-VNNI-INT16, OpenMP.
- Vulkan is not supported. ncnn builds its compute shaders through glslang,
  which is a git submodule and not part of the upstream release tarball used
  here. `-Dvulkan=true` is rejected at configure time.
- Threading uses ncnn's built-in thread pool.

Verified:

- `zig build` on Linux, plus compile-only cross builds for
  `x86_64-windows-gnu`, `aarch64-macos`, `aarch64-linux-gnu` and
  `arm-linux-gnueabihf`.
- `zig build test`: version match, options, `Input -> ReLU`, custom layer,
  data reader loading, pooled allocators, Mat layout/reshape/packing,
  ParamDict round trip, pixel round trip, drawing.
- `zig build test-upstream`: six upstream layer tests against this library.

Everything else in the wrapper is unverified.

## Development

```bash
zig build               # compile the library
zig build bindings      # generate and type-check the wrapper
zig build test          # wrapper smoke test
zig build test-upstream # upstream layer tests (kernels vs naive reference)
```

ncnn's C++ is built with `sanitize_c = .off`. Upstream relies on type punning
and pointer arithmetic that UBSan rejects.
