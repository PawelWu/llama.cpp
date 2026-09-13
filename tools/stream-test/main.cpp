// stream-test: minimal double-buffering test
// Demonstrates: H2D copy (transfer queue) overlapping with compute (compute queue).
//
// Model: pinned host buffer (weights) -> VRAM ring (2 slots).
// For each chunk i:
//   1. H2D copy chunk i -> VRAM slot A (transfer queue)
//   2. D2D copy slot A -> scratch (compute queue)  [proxy for real compute]
//   3. While (2) runs, H2D copy chunk i+1 -> VRAM slot B (transfer queue)
//   4. Swap A/B
//
// Usage: llama-stream-test [chunk_mb] [n_chunks]
//   chunk_mb  : size of each chunk in MB (default 256)
//   n_chunks  : number of chunks (default 4)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t.time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    size_t chunk_mb   = (argc > 1) ? (size_t) atoi(argv[1]) : 256;
    int    n_chunks   = (argc > 2) ? atoi(argv[2])         : 4;
    size_t chunk_size = chunk_mb * 1024 * 1024;

    printf("stream-test: chunk=%zu MB, n_chunks=%d\n", chunk_mb, n_chunks);

    // init Vulkan backend
    ggml_backend_t vk = ggml_backend_vk_init(0);
    if (!vk) {
        fprintf(stderr, "failed to init Vulkan backend\n");
        return 1;
    }

    // buffer types
    ggml_backend_buffer_type_t buft_dev  = ggml_backend_vk_buffer_type(0);   // VRAM
    ggml_backend_buffer_type_t buft_host = ggml_backend_vk_host_buffer_type(); // pinned RAM

    // ggml context (metadata only; tensor data lives in backend buffers, not here)
    std::vector<uint8_t> ctx_mem(1024 * 1024); // 1 MB for tensor metadata
    struct ggml_init_params params = {
        /*.mem_size =*/ ctx_mem.size(),
        /*.mem      =*/ ctx_mem.data(),
        /*.no_alloc =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // host buffer: n_chunks * chunk_size (all weights resident in pinned RAM)
    size_t host_size = (size_t) n_chunks * chunk_size;
    ggml_backend_buffer_t host_buf = ggml_backend_buft_alloc_buffer(buft_host, host_size);
    if (!host_buf) {
        fprintf(stderr, "failed to alloc host buffer (%zu MB)\n", host_size / (1024*1024));
        return 1;
    }
    // fill with a pattern so the D2D copy has real data to move
    uint8_t * host_base = (uint8_t *) ggml_backend_buffer_get_base(host_buf);
    for (size_t i = 0; i < host_size; i += 4096) {
        host_base[i] = (uint8_t) (i & 0xff);
    }

    // VRAM ring: 2 slots, each chunk_size
    size_t vram_slot_size = chunk_size;
    ggml_backend_buffer_t vram_a = ggml_backend_buft_alloc_buffer(buft_dev, vram_slot_size);
    ggml_backend_buffer_t vram_b = ggml_backend_buft_alloc_buffer(buft_dev, vram_slot_size);
    if (!vram_a || !vram_b) {
        fprintf(stderr, "failed to alloc VRAM buffers (%zu MB each)\n", vram_slot_size / (1024*1024));
        return 1;
    }

    // scratch buffer for the D2D copy (proxy compute)
    ggml_backend_buffer_t scratch = ggml_backend_buft_alloc_buffer(buft_dev, vram_slot_size);
    if (!scratch) {
        fprintf(stderr, "failed to alloc scratch buffer\n");
        return 1;
    }

    // create tensors (metadata only; data is in the backend buffers)
    // host tensor: one big tensor covering the whole host buffer
    struct ggml_tensor * t_host = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, host_size / 4);
    ggml_backend_tensor_alloc(host_buf, t_host, ggml_backend_buffer_get_base(host_buf));

    // VRAM slot tensors
    struct ggml_tensor * t_a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vram_slot_size / 4);
    ggml_backend_tensor_alloc(vram_a, t_a, ggml_backend_buffer_get_base(vram_a));

    struct ggml_tensor * t_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vram_slot_size / 4);
    ggml_backend_tensor_alloc(vram_b, t_b, ggml_backend_buffer_get_base(vram_b));

    struct ggml_tensor * t_scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vram_slot_size / 4);
    ggml_backend_tensor_alloc(scratch, t_scratch, ggml_backend_buffer_get_base(scratch));

    // warmup: one H2D + D2D to initialize the queues
    ggml_backend_tensor_set_async(vk, t_a, host_base, 0, chunk_size);
    ggml_backend_tensor_copy_async(vk, vk, t_a, t_scratch);
    ggml_backend_synchronize(vk);

    // --- benchmark: double-buffering ---
    // H2D(0) -> A before the loop. Then in iteration i:
    //   D2D(slot i%2) -> scratch  (compute chunk i, compute queue)
    //   H2D(i+1) -> slot (i+1)%2  (prefetch chunk i+1, transfer queue, overlaps with D2D)
    // Each chunk is H2D-copied exactly once.
    double t_start = now_ms();

    // prime the pipeline: H2D chunk 0 -> slot A
    ggml_backend_tensor_set_async(vk, t_a, host_base, 0, chunk_size);

    for (int i = 0; i < n_chunks; i++) {
        struct ggml_tensor * cur_t = (i % 2 == 0) ? t_a : t_b;
        struct ggml_tensor * nxt_t = (i % 2 == 0) ? t_b : t_a;

        // compute chunk i: D2D cur_t -> scratch (compute queue)
        ggml_backend_tensor_copy_async(vk, vk, cur_t, t_scratch);

        // prefetch chunk i+1 -> nxt_t (transfer queue), overlaps with the D2D above
        if (i + 1 < n_chunks) {
            size_t nxt_offset = (size_t) (i + 1) * chunk_size;
            ggml_backend_tensor_set_async(vk, nxt_t, host_base + nxt_offset, 0, chunk_size);
        }
    }

    ggml_backend_synchronize(vk);
    double t_end = now_ms();
    double total_ms = t_end - t_start;

    // --- baseline: no overlap (H2D then D2D, serialized) ---
    double t_base_start = now_ms();
    for (int i = 0; i < n_chunks; i++) {
        ggml_backend_buffer_t cur_buf = (i % 2 == 0) ? vram_a : vram_b;
        struct ggml_tensor * cur_t = (i % 2 == 0) ? t_a : t_b;
        size_t offset = (size_t) i * chunk_size;
        // H2D, wait
        ggml_backend_tensor_set_async(vk, cur_t, host_base + offset, 0, chunk_size);
        ggml_backend_synchronize(vk);
        // D2D, wait
        ggml_backend_tensor_copy_async(vk, vk, cur_t, t_scratch);
        ggml_backend_synchronize(vk);
    }
    ggml_backend_synchronize(vk);
    double t_base_end = now_ms();
    double base_ms = t_base_end - t_base_start;

    double total_mb = (double) n_chunks * chunk_mb;
    printf("\nresults:\n");
    printf("  double-buffered : %8.1f ms  (%.2f GB/s effective)\n", total_ms, total_mb / 1024.0 / (total_ms / 1000.0));
    printf("  serialized      : %8.1f ms  (%.2f GB/s effective)\n", base_ms,   total_mb / 1024.0 / (base_ms   / 1000.0));
    printf("  speedup         : %.2fx\n", base_ms / total_ms);

    // cleanup
    ggml_free(ctx);
    ggml_backend_buffer_free(host_buf);
    ggml_backend_buffer_free(vram_a);
    ggml_backend_buffer_free(vram_b);
    ggml_backend_buffer_free(scratch);
    ggml_backend_free(vk);

    return 0;
}
