# Chunked Decode NaN - Next Steps

## Problem
- Chunk 1 (i_start=0): valid hidden state values
- Chunk 2 (i_start=10): NaN hidden state values
- `t_hidden` buffer contains valid data after copy (verified via CPU readback)
- But when chunk 2's graph reads `t_hidden` as input, computation produces NaN

## Root cause hypothesis
The `t_hidden` tensor lives in a separately-allocated buffer (`ggml_backend_buft_alloc_buffer`).
The Vulkan backend scheduler may not properly bind/transition this external buffer for
shader reads during graph compute. The buffer state after `tensor_copy` is TRANSFER_DST,
but the compute shader needs SHADER_READ_ONLY.

## Options

### 1. Use `ggml_backend_sched` buffer instead of external buffer
Allocate `t_hidden` inside the scheduler's compute buffer (same buffer the graph uses).
This guarantees the Vulkan backend already knows how to read from it.
- How: use `ggml_backend_sched_alloc_buffer` or allocate in the same `ggml_backend_buffer_t`
  that the scheduler uses for compute.
- Risk: compute buffer is reused/cleared between graphs. Need to verify `t_hidden` survives.

### 2. Use `ggml_backend_graph_compute` with explicit buffer registration
Register the `t_hidden` buffer with the scheduler before compute so it's properly bound.
- How: call `ggml_backend_sched_register_buffer` (if exists) or use `ggml_backend_sched_set_tensor_buffer`
- Risk: API may not exist.

### 3. Copy hidden state INTO the compute buffer via a graph node
Instead of passing `t_hidden` as an external tensor, create a `ggml_set_param` / input tensor
inside the graph context that the scheduler manages. Copy data into it before compute.
- How: in `build_inp_embd`, create a new tensor in `ctx0`, mark as input, then copy data
  from `t_hidden` into it using `ggml_backend_tensor_copy` AFTER graph build but BEFORE compute.
- Risk: need access to the compute buffer's memory.

### 4. Use host-pinned CPU buffer + explicit H2D copy in graph
Allocate `t_hidden` in CPU pinned memory. In the graph, add an explicit copy node from
CPU to device. The scheduler handles the H2D transfer as part of the graph.
- How: create a `ggml_cont` or use `ggml_backend_tensor_copy` as a graph operation.
- Risk: adds latency per chunk.

### 5. Skip the external buffer entirely - use `t_layer_out` pointer directly
Keep a raw pointer to the previous chunk's `t_layer_out` data. Before the next chunk's
compute, copy that data into a tensor that lives in the compute buffer (allocated by scheduler).
- How: after chunk N, save `t_layer_out->data` pointer + nbytes. Before chunk N+1,
  allocate a tensor in the graph, copy data into it.
- Risk: `t_layer_out` data may be in a reused compute buffer slot.

### 6. Test with CPU-only PP (no Vulkan) to isolate backend issue
Run PP with `--pp-dev CPU` to see if chunked decode works without Vulkan.
If it works on CPU, the issue is Vulkan-specific buffer handling.
- How: change `--pp-dev` to CPU, remove `--pp-stream-mb` (not needed for CPU).
- Risk: slow, but confirms the hypothesis.

## Recommended order
1. **Option 6 first** (quick test, confirms Vulkan-specific issue)
2. **Option 1** (most likely fix - use scheduler-managed buffer)
3. **Option 3** (fallback if option 1 doesn't work)
