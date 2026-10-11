#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// pp backend policy for the scheduler: which ops follow the backend selected through
// ggml_backend_sched_set_offload_backend instead of following their inputs or their weights.
// only used from ggml-backend.cpp

#ifdef __cplusplus
extern "C++" {
#endif

int ggml_backend_pp_dev_op_batch_size(const struct ggml_tensor * tensor);

bool ggml_backend_pp_dev_force_flash_attn(ggml_backend_t backend, const struct ggml_tensor * tensor);
bool ggml_backend_pp_dev_force_rms_norm(ggml_backend_t backend, const struct ggml_tensor * tensor);
bool ggml_backend_pp_dev_force_weighted_op(ggml_backend_t backend, const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
