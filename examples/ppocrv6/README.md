# PP-OCRv6 text-line recognition

Small consumer example for `ncnn.zig`: recognizes one cropped text line with
PP-OCRv6 tiny and prints the text. The model, weights, CTC dictionary and a
sample text line are embedded in the binary, so there is nothing to download or
place on disk.

This example covers **recognition only**. It recognizes a cropped text line
image; full-page detection (DB post-processing, cropping) is intentionally out
of scope.

## Usage

```bash
zig build run
```

Output:

```
output: 48x6906
text: 绿洲仕格维花园公寓
score: 0.9868
```

`output: rows x cols` is the raw CTC head shape. The dictionary's first line
is the blank key, so `cols` equals the dictionary line count. The demo is also
a quick way to check that a model graph loads and runs through the bindings.

The example takes no arguments: the image, model, weights and dictionary are
all compiled in.

## Notes

- The embedded assets are `src/models/PP_OCRv6_tiny_rec.param`,
  `src/models/PP_OCRv6_tiny_rec.bin`, `src/models/ppocr_keys_v6_tiny.txt` and
  `src/assets/sample.png` (about 2.3 MB in total). They come from the
  [PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP) release
  archive and are covered by PaddleOCR's Apache-2.0 license.
- Inference runs with `NumThreads = 1` so timings are reproducible.
- The image is resized to height 48 with the aspect ratio preserved, then
  normalized to `[-1, 1]` (`127.5` mean/std), matching PaddleOCR's recognizer.
- Add `.x86_isa` / `.arm_isa` in `build.zig` when you need to trim the dispatch
  variants; the example forwards both options like the package does.
