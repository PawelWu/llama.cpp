# Phase 3 - KV transfer optimizations

## Phase 3.1 - Dedicated KV copy (used range only) [DONE]

### Problem

The PoC (pp-sep) used the checkpoint API (`llama_state_seq_get_data_ext` /
`llama_state_seq_set_data_ext`) to transfer KV state from the PP context to the decode
context. This goes through a host buffer and includes cell metadata (positions, seq_ids).

### Solution

Added a dedicated `llama_kv_cache_copy_from` function to the public API:

```c
// include/llama.h
LLAMA_API void llama_kv_cache_copy_from(llama_context * ctx_dst, const llama_context * ctx_src, uint32_t n_cells);
```

Implementation:
- `src/llama-context.cpp`: `llama_kv_cache_copy_from` free function. Uses `llama_get_memory()`
  to get the memory object, then `dynamic_cast` to handle both `llama_kv_cache` and
  `llama_memory_hybrid` (which wraps a `llama_kv_cache` via `get_mem_attn()`).
- `src/llama-kv-cache.cpp`: `copy_from` method. Iterates over layers, creates 3D views
  over the first `n_cells` cells of stream 0 for K and V tensors, sets buffer pointers
  manually (views have `buffer = NULL` by default), calls `ggml_backend_tensor_copy`.

### Results

| Metric | Checkpoint API | copy_from |
|--------|---------------|-----------|
| KV transfer time | 5.8 ms | 0.2 ms |
| Data copied | 50.7 MB (with metadata) | ~1.9 MB (K/V only, 26 cells) |

The `copy_from` path is ~30x faster because it skips the host round-trip and metadata.

### Files modified

- `include/llama.h` - new function declaration
- `src/llama-context.cpp` - new function implementation + `#include "llama-memory-hybrid.h"`
- `src/llama-kv-cache.h` - new `copy_from` method declaration
- `src/llama-kv-cache.cpp` - new `copy_from` method implementation
- `pocs/pp-sep/main.cpp` - replaced checkpoint API call with `llama_kv_cache_copy_from`

## Phase 3.2 - Async / Overlapped Copy [BLOCKED]

### Problem

`ggml_backend_tensor_copy` is synchronous. To overlap the KV copy with the start of
decode computation, we need an async version.

`ggml_backend_tensor_copy_async` exists in `ggml-backend.h` but requires explicit
`backend_src` and `backend_dst` parameters. The `ggml_backend_buffer` struct does not
expose a backend pointer - only a `buft` (buffer type) with a `device` member.

### Next steps

- Find a way to get the backend from a tensor's buffer (or pass the backends explicitly
  from the PoC, which knows which devices it used).
- Alternatively: skip the async copy for now. The 0.2 ms transfer time is already
  negligible compared to the 2+ second prefill and 3+ second decode.

## Phase 3.3 - Overlap weight streaming with prefill [TODO]

### Idea

While the PP context is doing prefill on CUDA0, start streaming the model weights
to the decode device (Vulkan1) in the background. This overlaps the weight transfer
with the prefill computation.

### Implementation sketch

1. After creating the decode context, start a background thread that copies the model
   weights from the PP context's buffers to the decode context's buffers.
2. The prefill runs on CUDA0 in the main thread.
3. When prefill finishes, wait for the weight streaming to complete.
4. Then do the KV transfer and start decode.

### Challenge

The model weights are stored in the `llama_model` object, which is shared between both
contexts (same model pointer). So the weights are already in the same memory - no
streaming needed. The "streaming" only applies if we use different model instances
or if the weights are on different devices.

For the current PoC setup (same model, different devices for KV), the weights are
already accessible from both contexts. No streaming needed.

## Phase 3.4 - Reuse the model weights [TODO]

### Idea

Instead of loading the model twice (once for PP, once for decode), load it once and
share the weights between both contexts.

### Current state

The PoC already shares the model between both contexts (same `llama_model` pointer).
The weights are loaded once and used by both contexts. No changes needed.

### Verification

Check that both contexts use the same model pointer. If so, Phase 3.4 is already done.
