# Chunked Decode - Root Cause Analysis

## Problem
Chunk 2+ produces NaN. t_hidden data is VALID before compute, but graph output is NaN.

## Root Cause
Vulkan backend cannot read t_hidden from a separate buffer during graph compute.
- CPU buffer: fails (Vulkan can't read CPU memory in compute shader)
- GPU buffer (separate): fails (not bound to compute pipeline)
- ggml_cont: fails (same issue - source is external buffer)
- stream_buf: weights work because they are MODEL TENSORS registered with backend

## Solution Approach
Create hidden state tensor INSIDE graph context (ctx0) so scheduler allocates it
in the compute buffer. Copy data into it AFTER sched_alloc_graph, BEFORE graph_compute.

Steps:
1. build_inp_embd: create tensor in ctx0, do NOT ggml_set_input (let scheduler alloc)
2. process_ubatch: after sched_alloc_graph, copy t_hidden data into the new tensor
3. graph_compute reads from the tensor in compute buffer (works like any other tensor)

## Key files
- src/llama-graph.cpp: build_inp_embd (line ~2363)
- src/llama-context.cpp: process_ubatch (line ~1363), chunked loop (line ~1890)

## Debug prints to remove before commit
- [process_ubatch], [chunked] BEFORE, [chunked] CPU, [chunked] t_layer_out GPU
- [build_inp_embd], [can_reuse], [eval_dbg]
