#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// pp backend policy (see --pp-dev): weight ops of prefill-sized batches run on a dedicated device
// instead of the device that holds the weights, which are copied into its compute buffer per split
namespace llama_pp_dev {

// point the scheduler at the backend built on `pp_backend` if `n_tokens` is large enough to pay for
// the weight copies; does nothing when pp_backend is NULL
void set_offload_backend(ggml_backend_sched_t sched, const std::vector<ggml_backend_t> & backends,
    ggml_backend_dev_t pp_backend, uint32_t n_tokens);

// value-based on purpose: an env var set to an empty string is still "set", and a batch `set VAR=`
// that keeps the trailing space is easy to get wrong. GGML_PP_NORESV_FIX=1 keeps the old behavior
bool rearm_after_reserve();

// build the tensor buft overrides that keep the weights of every layer running on `pp_dev` inside
// that device's own buffer type, followed by `user_overrides`. out_strings must stay alive next to
// out_overrides - it holds the patterns they point at. returns false when nothing was built (no
// pp device, or no layer on it) and true when the caller must install out_overrides.data()
bool build_resident_overrides(ggml_backend_dev_t pp_dev, int n_layer_all,
    const std::function<ggml_backend_dev_t(int)> & layer_dev,
    const llama_model_tensor_buft_override * user_overrides,
    std::vector<std::string> & out_strings,
    std::vector<llama_model_tensor_buft_override> & out_overrides);

} // namespace llama_pp_dev
