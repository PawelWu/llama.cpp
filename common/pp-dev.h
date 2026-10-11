#pragma once

#include "arg.h"
#include "speculative.h"

#include <vector>

// common-side pieces of the --pp-dev policy (the llama-side policy lives in src/llama-pp-dev.h)

// the options that --pp-dev adds to the arg parser, in declaration order
std::vector<common_arg> common_pp_dev_get_args();

// logs the pp device, warns about the per-batch weight copies and sets pp_dev_resident when asked
void common_pp_dev_apply_model_params(const common_params & params, llama_model_params & mparams);

// n_batch for a draft context, -1 if it keeps its current value. block-diffusion drafts are capped
// by GGML_DRAFT_B so their flash-attn working set stays small on a dGPU that also stages pp weights
int32_t common_pp_dev_draft_batch_cap(const std::vector<enum common_speculative_type> & types, int32_t n_batch);
