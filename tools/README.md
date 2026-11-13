# Tools

## layers.py

Suggests a `.layers` allowlist for ncnn-zig from ncnn `.param` files.

```bash
python3 tools/layers.py model.param
python3 tools/layers.py det.param rec.param cls.param
```

Pass one or more files; the result is their union. The output is one line:

```zig
.layers = "Input,Convolution,...,Bias,Scale,Cast",
```

Copy it into:

```zig
const ncnn_dep = b.dependency("ncnn_zig", .{
    .target = target,
    .optimize = optimize,
    .layers = "...",
});
```

### What is included

- the layer types from the `.param` files;
- fused activations (`9=`) mapped to `ReLU`, `Clip`, `Sigmoid`, `Mish` or
  `HardSwish`;
- the default helper set: `Packing`, `Flatten`, `Padding`, `Interp`, `Crop`,
  `Cast`, `Bias`, `Scale`.

The result targets a runnable build, not a minimal one. Remove helper layers
you do not need and test the model. `--minimal` omits the helper set and warns
about fp16/bf16 weights and Mat/pixel APIs.

`--sources PATH` points at a different `src/sources.zig` (default: this
checkout).

### Notes

- ncnn-zig still expands static dependencies such as `Convolution -> Gemm`;
  they do not need to be in the string.
- Custom layer types are reported but not added; register them at runtime.
- fp16/bf16 weights load through `Cast`; it is in the default helpers.
- Some helper needs are application-specific (for example
  `subtractMeanNormalize` needs `Bias` and `Scale`), so `--minimal` requires
  testing the model.
- Only existing `.param` files are read; generated or embedded models are not
  resolved.

## check_deps.zig

Not user-facing. `zig build check-deps` scans upstream layer sources and
verifies `runtime_dependencies` in `src/sources.zig`.
