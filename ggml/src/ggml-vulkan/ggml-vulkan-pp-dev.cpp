#include "ggml-vulkan-common.h"

#include <cstdlib>

// host-stage decision for the MULTI_DEVICE path of ggml_vk_buffer_copy: whether the mapped host
// pointer of the source buffer may be read by the CPU, instead of staging through the GPU.
// uncached host-visible UMA memory reads back at about 100 MB/s on the CPU, so only cached (pinned)
// sources take the direct path - they also leave the src queue alone, which matters when several
// devices copy at once. off by default; GGML_VK_HOST_STAGE=1 turns it on
bool ggml_vk_pp_dev_host_stage(const vk_buffer & src) {
    static const bool host_stage = []() {
        const char * env = std::getenv("GGML_VK_HOST_STAGE");
        return env != NULL && std::atoi(env) != 0;
    }();

    if (!host_stage) {
        return false;
    }

    const bool src_cached  = !!(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    const bool src_visible = !!(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible);

    return src_cached && src_visible;
}
