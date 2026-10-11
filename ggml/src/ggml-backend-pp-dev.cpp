#include "ggml-backend-pp-dev.h"

#include <cstdlib>

// batch size of an op, used to tell prefill-sized work apart from decode-sized work:
// attention and norms scale with the number of rows, mul_mat with its second dimension
int ggml_backend_pp_dev_op_batch_size(const struct ggml_tensor * tensor) {
    switch (tensor->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return (int) tensor->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return (int) tensor->ne[2];
        default:
            return (int) ggml_nrows(tensor);
    }
}

// prefill-sized batches only. decode keeps running on the backend that owns the weights,
// where the batch is too small for a weight copy to pay for itself
static const int pp_dev_min_batch = 32;

bool ggml_backend_pp_dev_force_flash_attn(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    if (tensor->op != GGML_OP_FLASH_ATTN_EXT || ggml_backend_pp_dev_op_batch_size(tensor) < pp_dev_min_batch) {
        return false;
    }
    return ggml_backend_supports_op(backend, tensor);
}

// rms norm has no weight source (eps is an op param), so it follows its input tensor onto the
// device holding that tensor. for prefill-sized batches its output would then be copied once per
// consumer on the pp backend, so run the op there instead
bool ggml_backend_pp_dev_force_rms_norm(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    // GGML_PP_NO_NORM=1 leaves rms norm with its input, to A/B the split count on one binary
    static const bool force_norm = std::getenv("GGML_PP_NO_NORM") == NULL;

    if (!force_norm || tensor->op != GGML_OP_RMS_NORM || ggml_backend_pp_dev_op_batch_size(tensor) < pp_dev_min_batch) {
        return false;
    }
    return ggml_backend_supports_op(backend, tensor);
}

// ops that read weights: for prefill-sized batches the weights are copied into the compute buffer
// of the pp backend once per graph split and reused by ggml-alloc, which is cheaper than keeping
// the whole prefill on the device that owns the weights
bool ggml_backend_pp_dev_force_weighted_op(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    if (ggml_backend_pp_dev_op_batch_size(tensor) < pp_dev_min_batch) {
        return false;
    }
    return ggml_backend_supports_op(backend, tensor) && ggml_backend_offload_op(backend, tensor);
}
