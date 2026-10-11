#include "pp-dev.h"

#include "log.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

std::vector<common_arg> common_pp_dev_get_args() {
    std::vector<common_arg> args;

    args.push_back(common_arg(
        {"--pp-dev"}, "DEVICE",
        "device that should run prefill (e.g. CUDA0)\n"
        "weight ops of prefill batches (batch size >= 32) run on it,\n"
        "the weights are copied to this device on every such batch, wherever they are stored\n"
        "decode (n_tokens = 1) is not affected",
        [](common_params & params, const std::string & value) {
            auto devices = parse_device_list(value);
            // parse_device_list pushes nullptr at back so devices is length 2 for single device.
            if (devices.size() > 2) {
                throw std::invalid_argument("only one device may be specified for --pp-dev");
            }
            params.pp_dev = devices.front();
        }
    ).set_env("LLAMA_ARG_PP_DEV"));

    args.push_back(common_arg(
        {"--pp-dev-resident"},
        "keep the weights of the layers that --pp-dev owns in its own VRAM\n"
        "the layers are the ones the layer split assigns to the pp device (-ts/--tensor-split)\n"
        "so they are read in place instead of being copied on every prefill batch",
        [](common_params & params) {
            params.pp_dev_resident = true;
        }
    ).set_env("LLAMA_ARG_PP_DEV_RESIDENT"));

    return args;
}

void common_pp_dev_apply_model_params(const common_params & params, llama_model_params & mparams) {
    if (params.pp_dev == nullptr) {
        return;
    }

    COM_INF("pp_device = %s (%s)\n", ggml_backend_dev_name(params.pp_dev), ggml_backend_dev_description(params.pp_dev));

    if (params.n_gpu_layers != 0) {
        COM_WRN("%s", "--pp-dev copies the weights of offloaded layers to the prefill device on every prefill batch\n");
    }

    if (params.pp_dev_resident) {
        mparams.pp_dev_resident = params.pp_dev;
    }
}

int32_t common_pp_dev_draft_batch_cap(const std::vector<enum common_speculative_type> & types, int32_t n_batch) {
    const bool has_block_draft = std::any_of(
        types.begin(), types.end(), [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });

    if (!has_block_draft) {
        return -1;
    }

    const char * e = std::getenv("GGML_DRAFT_B");
    const int32_t cap = e ? std::atoi(e) : 0;

    if (cap <= 0 || n_batch <= cap) {
        return -1;
    }

    LOG_INF("%s: clamping draft context n_batch to %d (GGML_DRAFT_B), was %d\n", __func__, (int) cap, (int) n_batch);

    return cap;
}
