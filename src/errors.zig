//! Error set shared by the wrapper.

/// Failures reported by the C API.
pub const Error = error{
    // Allocation failed.
    OutOfMemory,
    // ncnn_net_create returned null.
    CreateNetFailed,
    // ncnn_extractor_create returned null.
    CreateExtractorFailed,
    // ncnn_option_create returned null.
    CreateOptionFailed,
    // ncnn_mat_create returned null.
    CreateMatFailed,
    // ncnn_paramdict_create returned null.
    CreateParamDictFailed,
    // ncnn_layer_create returned null.
    CreateLayerFailed,
    // ncnn_datareader_create returned null.
    CreateDataReaderFailed,
    // ncnn_modelbin_create returned null.
    CreateModelBinFailed,
    // ncnn_allocator_create returned null.
    CreateAllocatorFailed,
    // load_param returned non-zero.
    LoadParamFailed,
    // load_model returned non-zero.
    LoadModelFailed,
    // extractor input returned non-zero.
    InputFailed,
    // extractor extract returned non-zero.
    ExtractFailed,
    // Path or blob name does not fit the stack buffer.
    PathTooLong,
    // Mat is not plain f32 (bf16, int8 or packed data).
    NotFloat32,
    // Mat dims cannot be read as one flat slice.
    UnsupportedDims,
    // Blob index is past the end.
    IndexOutOfBounds,
    // Header version differs from the linked runtime.
    VersionMismatch,
    // Pixel buffer is too small for the requested operation.
    InvalidPixelBuffer,
};
