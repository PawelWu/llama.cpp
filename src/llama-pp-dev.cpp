#include "llama-pp-dev.h"

#include "llama-impl.h"

#include <cstdlib>

// smallest ubatch that is still worth running on the pp backend. the forced offload copies the
// weights of every offloaded op to the pp backend, which costs the same per ubatch no matter its
// size, so only prefill-sized batches can pay for it. decode batches (1 token, or a few with
// speculative decoding) belong on the device that owns the weights
static constexpr uint32_t PP_BACKEND_MIN_BATCH = 32;

namespace llama_pp_dev {

void set_offload_backend(ggml_backend_sched_t sched, const std::vector<ggml_backend_t> & backends,
    ggml_backend_dev_t pp_backend, uint32_t n_tokens) {
    if (!pp_backend) {
        return;
    }

    for (auto & backend : backends) {
        if (ggml_backend_get_device(backend) == pp_backend) {
            ggml_backend_sched_set_offload_backend(sched, n_tokens >= PP_BACKEND_MIN_BATCH ? backend : nullptr);
            return;
        }
    }

    LLAMA_LOG_WARN("%s: %s is not used by this context, ignoring\n", __func__, ggml_backend_dev_name(pp_backend));
}

// see the declaration comment in llama-pp-dev.h
bool build_resident_overrides(ggml_backend_dev_t pp_dev, int n_layer_all,
    const std::function<ggml_backend_dev_t(int)> & layer_dev,
    const llama_model_tensor_buft_override * user_overrides,
    std::vector<std::string> & out_strings,
    std::vector<llama_model_tensor_buft_override> & out_overrides) {

    if (!pp_dev) {
        return false;
    }

    size_t n_resident = 0;
    for (int il = 0; il < n_layer_all; ++il) {
        if (layer_dev(il) == pp_dev) {
            n_resident++;
        }
    }

    if (n_resident == 0) {
        LLAMA_LOG_WARN("%s: no layer is assigned to device %s, no weights kept resident\n",
            __func__, ggml_backend_dev_name(pp_dev));
        return false;
    }

    auto * buft = ggml_backend_dev_buffer_type(pp_dev);

    // the overrides point into out_strings, so fill it first and only then take the pointers
    out_strings.clear();
    out_strings.reserve(n_resident);
    for (int il = 0; il < n_layer_all; ++il) {
        if (layer_dev(il) == pp_dev) {
            out_strings.push_back(format("blk\\.%d\\.", il));
        }
    }

    out_overrides.clear();
    out_overrides.reserve(n_resident + (user_overrides != nullptr ? n_layer_all : 0) + 1);
    for (const auto & pattern : out_strings) {
        out_overrides.push_back({ pattern.c_str(), buft });
    }

    // the user overrides come after, so they still apply to all other tensors
    if (user_overrides != nullptr) {
        for (const llama_model_tensor_buft_override * o = user_overrides; o->pattern != nullptr; ++o) {
            out_overrides.push_back(*o);
        }
    }
    out_overrides.push_back({ nullptr, nullptr });

    LLAMA_LOG_INFO("%s: keeping the weights of %zu layers in %s\n", __func__, n_resident,
        ggml_backend_buft_name(buft));

    return true;
}

bool rearm_after_reserve() {
    const char * no_resv_fix = getenv("GGML_PP_NORESV_FIX");
    return no_resv_fix == NULL || no_resv_fix[0] != '1';
}

} // namespace llama_pp_dev
