#pragma once

#include "arg.h"

#include <vector>

// common-side pieces of the --pp-dev policy (the llama-side policy lives in src/llama-pp-dev.h)

// the options that --pp-dev adds to the arg parser, in declaration order
std::vector<common_arg> common_pp_dev_get_args();

// logs the pp device, warns about the per-batch weight copies and sets pp_dev_resident when asked
void common_pp_dev_apply_model_params(const common_params & params, llama_model_params & mparams);

