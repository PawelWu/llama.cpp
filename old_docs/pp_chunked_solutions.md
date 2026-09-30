# Chunked Decode - Possible Solutions

## Confirmed Facts
- Chunk 1 (i_start=0): valid output
- t_hidden data: VALID before chunk 2 compute
- t_inp_embd data: VALID after copy (first time only)
- Chunk 2 (i_start=10): NaN output from graph compute
- NOT a buffer issue (tried CPU, GPU, ctx0 tensor)
- NOT a copy issue (data confirmed valid)

## Possible Root Causes (to test in order)

### 1. KV cache not set up for chunked layers
The attention in chunk 2 needs KV from chunk 1. If KV cache slots are not
properly allocated/initialized for the PP context, attention produces NaN.
TEST: Check if mctx->apply() is called correctly for each chunk.
      Check if KV cache has valid data for positions 0..9 when chunk 2 runs.

### 2. inp_pos (position embeddings) wrong for chunked layers
The RoPE positions must be correct. If inp_pos is [0..7] for all chunks
instead of [10..17] for chunk 2, the attention would be wrong.
TEST: Log inp_pos values for each chunk.

### 3. build_attn_inp_kv() not handling chunked case
The attention input (K, V cache pointers) might not be set up correctly
for layers that start mid-sequence.
TEST: Log the KV cache pointers for chunk 2.

### 4. The graph is using token embeddings instead of t_hidden
Even though build_inp_embd returns t_hidden_graph, the gemma model might
be using tok_embd directly somewhere.
TEST: Add assert in gemma.cpp that inpL != tok_embd when i_start > 0.

### 5. n_ctx / seq_len mismatch
The attention mask or seq_len might be wrong for chunked layers.
TEST: Log n_ctx and seq_len for each chunk.

### 6. The t_inp_embd tensor is being freed/reused by scheduler
Despite ggml_set_input, the scheduler might still reuse the memory.
TEST: Check if t_inp_embd->data pointer changes between chunks.

## Current Debug State
- [copy] shows: first copy valid, subsequent copies NaN (because t_layer_out is NaN)
- The REAL issue: chunk 2 graph compute produces NaN despite valid input

## Next Test Priority
1. Log inp_pos for each chunk (quick test)
2. Check KV cache state (medium test)
3. Add assert in gemma.cpp (quick test)
