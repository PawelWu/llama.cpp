# Prefill/Decode device split (RTX prefill, RX decode)

Goal: run prompt processing (prefill) on the NVIDIA RTX (CUDA, 6 GB VRAM) and
token generation (decode) on the AMD iGPU (Vulkan, 32 GB shared RAM), in a
single inference session. The KV cache is NOT shared in RAM - it is written on
the RTX during prefill and copied to the RX after prefill, before decode.

## Status (updated after Phase 0-1)

- Phase 0 (hardware verification): done, see 0.1-0.3.
- Phase 1 (KV transfer + harness + correctness): done. The PoC works end-to-end
  and the split output matches the same-device control run token for token.
  The transfer uses the existing checkpointing API (host buffer round-trip),
  not a new core API - see the deviation note in 1.1.
- Phase 2 (usability of the pp-sep tool): done. The tool now applies the model's
  chat template by default (--raw for the old raw-prompt path), supports standard
  sampling (--temp/--top-k/--top-p/--min-p/--seed), and -c for context size.
  Verified on Qwen3.5-9B: the chat-template path triggers the model's thinking
  mode (first token = thinking start), the raw path gives the direct answer; both
  transfer the KV correctly and decode correctly. See the deviation note in 2.1.
- Phase 3 (optimizations): not started (optional, only if Phase 1-2 are slow).
- Phase 4 (RTX weight streaming / double buffering): future, not started -
  recorded for later, see the note under Phase 4.

Environment notes:
- Test model: /d/shared/ai-models/Qwen3.5-9B-Q4_K_M.gguf (5.3 GB).
- Build: Release, CUDA + Vulkan, Ninja generator. The build needs the MSVC
  environment; helper script build_pp.bat at the repo root calls vcvars64.bat
  then ninja (usage: `cmd //c build_pp.bat pp-sep llama-cli`).
- Unrelated finding: `llama-completion` segfaults on this build before the log
  file is even opened (i.e. before any context is created). It does not use
  ctx_other/speculative, so it is not connected to the local modifications in
  src/llama-context.cpp (custom MTP draft backend selection, verified intact
  after the upstream pull). Treated as a separate upstream tool bug.
- Use `--log-file d:/tmp/...` for tool runs; do not run interactive llama-cli
  from a non-TTY without `--no-interaction` (it loops and eats RAM).

## 1. Target scenario

| phase   | device            | weights                          | KV cache                |
|---------|-------------------|----------------------------------|-------------------------|
| prefill | NVIDIA RTX (CUDA) | streamed from RAM (n_gpu_layers small) | on RTX VRAM (offload_kqv = true) |
| decode  | AMD iGPU (Vulkan) | resident (n_gpu_layers = all)    | on RX shared RAM (offload_kqv = true) |

After prefill finishes on the RTX, the KV cache (K + V for every layer, for the
prompt length) is copied from the RTX to the RX. Decode then runs on the RX,
reading the weights and the KV cache from its own 32 GB shared RAM.

Why this split: the RTX has only 6 GB, so it cannot hold the whole model - it
streams weights from RAM (standard offload). The RX has 32 GB of shared RAM, so
it can hold all weights plus the full KV cache. Prefill is compute-heavy and
benefits from the RTX's CUDA cores; decode is memory-bandwidth-bound and the
RX's large shared memory is well suited to it.

## 2. Current state of the codebase

What already exists:

- Per-layer device assignment: `llama_model::load_tensors` (src/llama-model.cpp,
  ~line 1420) assigns each layer to a device based on `n_gpu_layers`,
  `tensor_split`, `main_gpu`, `split_mode`. Layers below `i_gpu_start` stay on
  the CPU (streamed); the rest are spread over the GPU devices by the split.
- KV cache placement: `llama_kv_cache` constructor (src/llama-kv-cache.cpp,
  ~line 200) allocates the K/V tensors on `model.dev_layer(il)` when
  `offload_kqv` is true, or on the CPU buffer type when false.
- Cross-device tensor copy: `ggml_backend_tensor_copy` (ggml/src/ggml-backend.cpp,
  line 488) already handles the case where src and dst are on different
  devices - if neither buffer is host it falls back to a host staging buffer
  (malloc + `ggml_backend_tensor_get` + `ggml_backend_tensor_set` + free).
  There is also `ggml_backend_tensor_copy_async` (line 511).
- Two-context coordination: speculative decoding sets `cparams.ctx_other`
  (common/speculative.cpp line 2538) so the draft context's KV cache shares the
  cell metadata (`v_cells_impl`) with the target context. This shares metadata
  only, not the K/V data - it is not directly reusable for our data copy, but it
  is a useful reference for the "two contexts that coordinate" pattern.
- In-cache sequence copy: `llama_kv_cache::seq_cp` (src/llama-kv-cache.cpp line
  452) copies K/V data between streams of the SAME cache and updates the cell
  metadata. This is the closest existing reference for what we need to do across
  two different caches.

What does NOT exist (the gap we must fill):

- No way to run prefill on one device and decode on another within a single
  context. The graph is built once per unique parameter set
  (`llama_context::process_ubatch`, src/llama-context.cpp ~line 1350) and the
  scheduler assigns ops to devices based on the model's fixed per-layer device
  assignment. There is no per-phase device switching.
- No way to copy the KV cache (K/V data + cell metadata) from one context to
  another. `seq_cp` only works within a single cache.

## 3. Chosen approach

Use two separate `llama_context` objects, each backed by its own `llama_model`
instance (the model is loaded twice, once per device assignment):

- `ctx_prefill` / `model_prefill`: device assignment for the RTX
  (`n_gpu_layers` = a small number that fits in 6 GB, `tensor_split` targeting
  the RTX, `offload_kqv = true` so the KV cache lands on the RTX VRAM).
- `ctx_decode` / `model_decode`: device assignment for the RX
  (`n_gpu_layers = all`, `tensor_split` targeting the RX, `offload_kqv = true`
  so the KV cache lands on the RX shared RAM).

Flow:
1. `llama_decode(ctx_prefill, prompt_batch)` - prefill on the RTX.
2. Copy the KV cache from `ctx_prefill` to `ctx_decode` (the new function,
   task 1.1). This is a cross-device copy (RTX VRAM -> RX shared RAM) handled by
   `ggml_backend_tensor_copy` per layer/stream.
3. `llama_decode(ctx_decode, token_batch)` - decode on the RX, continuing from
   the copied KV state.

The two contexts use the same model (same hparams), so the K/V tensor layout is
identical and `ggml_are_same_layout` passes for the copy.

Why two contexts and not one: a single context has a single model with a single
per-layer device assignment. The prefill and decode phases need different weight
placements (RTX streams a few layers, RX holds all layers), so they cannot share
one model object. Two contexts is the composable, minimal-change solution.

Note on RAM: loading the model twice means the weights are held twice in system
RAM (the RTX's non-offloaded layers are streamed from RAM, and the RX's layers
live in shared RAM = system RAM). For a 7B Q4 model (~4 GB) this is ~8 GB total,
comfortable within 32 GB. For much larger models this doubles the RAM footprint
- see risks.

## 4. Detailed tasks

### Phase 0 - verify the hardware setup (no code changes) [DONE]

- [x] 0.1 Confirm both backends are detected. Build with CUDA + Vulkan enabled,
      run a tool (e.g. `llama-bench` or `llama-cli`) and check the log lists both
      the NVIDIA RTX (CUDA) and the AMD iGPU (Vulkan) as available devices.
      Check `ggml_backend_dev_count()` / the device enumeration log.

      Done: `pp-sep --list-devs` shows 4 devices:
      - [0] GPU  CUDA0    NVIDIA GeForce RTX 4050 Laptop GPU (6140 MB)
      - [1] GPU  Vulkan0  NVIDIA GeForce RTX 4050 Laptop GPU
      - [2] IGPU Vulkan1  AMD Radeon 760M Graphics (33694 MB shared)
      - [3] CPU  CPU      AMD Ryzen 5 8645HS
- [x] 0.2 Confirm the model fits on the RX (32 GB shared RAM) with all layers
      offloaded, and does NOT fit on the RTX (6 GB) - i.e. the RTX needs
      `n_gpu_layers` set to a small value. Estimate the per-layer weight size and
      pick an `n_gpu_layers` for the RTX that leaves headroom for the KV cache
      in the 6 GB.

      Done: test model is Qwen3.5-9B-Q4_K_M.gguf (5.3 GB, /d/shared/ai-models/).
      RTX works with `--pp-ngl 2` (~0.6 GB weights in VRAM, plenty of headroom).
      Decode model with n_gpu_layers = -1 fits comfortably in the 32 GB iGPU.
- [x] 0.3 Baseline: run a single-context inference (all on the RX, or all on the
      RTX with streaming) and record the output text + timing. This is the
      reference to compare against after the split is implemented.

      Done: single-device reference run (both phases on Vulkan1, see 1.3).
      Decode on the iGPU: ~11.4-12.1 t/s for this model.

### Phase 1 - core: cross-context KV cache copy + test harness [DONE, with deviation]

- [x] 1.1 Implement the cross-context KV cache copy. This is the main new
      infrastructure. Add a method to `llama_kv_cache` (src/llama-kv-cache.h /
      .cpp) and a public API in include/llama.h, e.g.:

        ```
        // copy the KV cache (K/V data + cell metadata) from src to dst
        // src and dst must be caches of the same model (same hparams)
        void llama_kv_cache::copy_from(const llama_kv_cache * src);
        LLAMA_API void llama_memory_copy_from(llama_memory_t dst, llama_memory_t src);
        ```

      Implementation outline (mirror `seq_cp`, src/llama-kv-cache.cpp line 452, but
      across two cache objects):
      1. For each KV layer `ikv` in `layers` (use `get_layer_ids()` /
         `map_layer_ids`), for each stream `s` in `[0, n_stream)`:
         - `ggml_backend_tensor_copy(layers[ikv].k_stream[s], src->layers[ikv].k_stream[s])`
         - `ggml_backend_tensor_copy(layers[ikv].v_stream[s], src->layers[ikv].v_stream[s])`
         `ggml_backend_tensor_copy` (ggml/src/ggml-backend.cpp line 488) handles the
         cross-device transfer (RTX VRAM -> RX shared RAM) via host staging.
      2. Copy the cell metadata: for each stream, copy `v_cells[s]` (positions,
         sequence ids, shifts, ext) from `src->v_cells[s]` to `v_cells[s]`, and set
         `v_heads[s] = src->v_heads[s]`. This makes the destination cache continue
         from the same position as the source.
      3. Synchronize both backends before/after the copy (the copy is blocking;
         see `ggml_backend_tensor_copy_async` line 511 if we later want async).
      4. Guard: assert the two caches have the same `n_layer`, `n_stream`, and
         K/V layout (`ggml_are_same_layout`).

      Keep it minimal: copy the full K/V tensors for the used streams. Copying only
      the used range (a sub-view up to the prompt length) is an optimization for
      Phase 3.

      DEVIATION: no new core API was added. The PoC reuses the existing
      checkpointing API instead: `llama_state_seq_get_data_ext` / `llama_state_seq_set_data_ext`
      with a host buffer. A save+load round-trip preserves the exact sequence state
      (cells + positions), so the decode context continues from where the prompt
      left off. Zero changes to src/ for the transfer itself. If a dedicated
      `llama_memory_copy_from` API is still wanted (e.g. to avoid the host
      round-trip), implement it in Phase 3 alongside the async copy.

- [x] 1.2 Create a test harness. A new small tool (e.g. tools/llama-pp-sep/main.cpp,
      or a test in an existing tool) that:
      1. Loads the model twice:
         - `model_prefill` = `llama_model_load_from_file(path, mparams_prefill)`
           with `mparams_prefill.n_gpu_layers = <small>`, `tensor_split` for the RTX,
           `main_gpu` = the RTX index.
         - `model_decode` = `llama_model_load_from_file(path, mparams_decode)`
           with `mparams_decode.n_gpu_layers = -1` (all), `tensor_split` for the RX,
           `main_gpu` = the RX index.
      2. Creates two contexts:
         - `ctx_prefill` = `llama_init_from_model(model_prefill, cparams_prefill)`
           with `cparams_prefill.offload_kqv = true`.
         - `ctx_decode` = `llama_init_from_model(model_decode, cparams_decode)`
           with `cparams_decode.offload_kqv = true`.
      3. Runs prefill: `llama_decode(ctx_prefill, prompt_batch)`.
      4. Copies the KV: `llama_memory_copy_from(llama_get_memory(ctx_decode), llama_get_memory(ctx_prefill))`.
      5. Runs decode: loop `llama_decode(ctx_decode, token_batch)` for N tokens,
         sampling with a shared `llama_sampler`.
      6. Prints the generated text.

      Reference for the two-context setup: common/speculative.cpp
      `common_speculative_init_result` (line ~2508) shows how to load a second
      model and create a second context with `ctx_other`. We do NOT set
      `ctx_other` here (we want separate K/V data, not shared metadata) - we copy
      the data explicitly instead.

      Done: `pocs/pp-sep/main.cpp` (registered in pocs/CMakeLists.txt). Loads the
      model twice, each pinned to a single device via `mparams.devices` +
      `split_mode = LLAMA_SPLIT_MODE_NONE`. CLI: `--pp-dev <name>`, `--dec-dev
      <name>`, `--pp-ngl K`, `-n N`, `--list-devs`. Greedy sampler, raw prompt
      (no chat template), prints per-phase timings.

- [x] 1.3 Verify correctness. Run the harness with a fixed prompt + seed and
      compare the generated text against the Phase 0.3 single-context baseline.
      They should match (modulo any minor non-determinism between the CUDA and
      Vulkan backends). If they diverge, check the cell metadata copy (positions /
      sequence ids) first - a wrong position makes the decode read the wrong KV
      row.

      Done: compared two runs with the same prompt (13 tokens) and -n 32:
      - split run:    PP on CUDA0 (pp-ngl 2), decode on Vulkan1
      - control run:  PP on Vulkan1, decode on Vulkan1 (same device, fake split)
      Both produced the identical 32-token output (first token 16, "... Paris
      ... The capital of France is **Paris** ..."), so the KV transfer and the
      decode continuation are correct. Any divergence between the split run and
      a pure single-device run would come from CUDA vs Vulkan logit differences
      only.

      Measured (Qwen3.5-9B-Q4_K_M, 13-token prompt, n_ctx = 45):
      - PP prefill on CUDA0: 13 tokens in ~1080 ms (includes first-run warmup)
      - KV transfer: 50.7 MB in 5.8 ms (~8.7 GB/s, host staging)
      - decode on Vulkan1: 32 tokens in ~2650 ms (~12.1 t/s)

### Phase 2 - orchestration and integration

- [x] 2.1 Add configuration. Device selection is by NAME, not index: `--pp-dev
      <name>` and `--dec-dev <name>`, where <name> is the name shown by device
      enumeration (e.g. `CUDA0`, `Vulkan0`, `Vulkan1`, `CPU`). Implemented in the
      tool: `find_device_by_name` in pocs/pp-sep/main.cpp matches
      `ggml_backend_dev_name(dev)` against the user string, and `--list-devs`
      prints the available names.

      DEVIATION: the split-specific args were NOT added to `common_params` /
      common/arg.cpp. Two reasons: (1) the modern `llama-cli` is a client that
      connects to (or spawns) `llama-server`, so the real inference loop is in the
      server, which is complex and multi-slot - integrating the split there is very
      invasive; (2) `common_params_parse` rejects unknown args and its internal
      parser (`common_params_parse_ex`) is static, so adding `--pp-dev`/`--dec-dev`
      to the shared arg infrastructure means editing arg.cpp plus a new
      `llama_example` - invasive for an experimental feature. So the tool stays
      self-contained: it parses `--pp-dev`/`--dec-dev`/`--pp-ngl` locally and does
      not pollute the shared arg list. If the split is later promoted into the
      server, move the args to common_params then.
- [x] 2.2 Integrate the flow into the main loop. The tool runs the full sequence:
      prefill the prompt on `ctx_pp`, sample the first token, copy the KV to
      `ctx_dec`, then run the decode loop on `ctx_dec` for the remaining tokens.
      The phase boundary is the prompt batch (n_tokens > 1) vs the single-token
      decode batches (n_tokens == 1) - the same distinction `process_ubatch` uses.
- [x] 2.3 Carry the sampler state. One `llama_sampler` (a sampler chain) is used
      for both phases: the first token is sampled from the prefill logits, then
      the decode loop samples from the decode logits with the same chain object,
      so the sampling state is consistent across the boundary.
- [x] 2.4 Handle context-length / memory sizing. Both contexts are sized with
      `n_ctx = prompt + n_predict` by default, overridable with `-c N`. (Sizing
      `ctx_pp` for the prompt only, smaller than `ctx_dec`, is a possible
      optimization but not needed for correctness; the RTX's 6 GB is kept small by
      `--pp-ngl` limiting the offloaded layers.)
- [x] 2.5 Make the tool usable for real models. Apply the model's chat template
      by default (system + user messages) so chat models get the correct framing;
      `--raw` gives the old raw-prompt path. Add standard sampling (`--temp`
      default 0 = greedy, `--top-k`, `--top-p`, `--min-p`, `--seed`) and `-c` for
      context size. When the template is used, tokenization interprets the
      template's special tokens (special = true, no extra BOS).

      Verified (Qwen3.5-9B-Q4_K_M, --pp-dev CUDA0 --dec-dev Vulkan1 --pp-ngl 2):
      - chat-template path (default): 26-token prompt, first token 248068 (the
        model's thinking-start token), output begins "Thinking Process: 1.
        **Analyze the Request:** ..." - correct, the template puts the model in
        its thinking mode. KV transfer 51.1 MB in 7.6 ms (~6.7 GB/s).
      - raw path (--raw): 13-token prompt, first token 271, output "The capital
        of France is **Paris**." - the old behavior, still correct. KV transfer
        50.7 MB in 6.7 ms (~7.5 GB/s).

### Phase 3 - optimization (optional, only if Phase 1-2 are correct and slow)

- [ ] 3.1 Copy only the used KV range. Instead of copying the full K/V tensors
      (sized to the max context), copy a sub-view up to the prompt length. Build
      a view tensor (`ggml_view_1d` / `ggml_view`) over the used rows and copy that.
      This reduces the transfer size for short prompts.
- [ ] 3.2 Async / overlapped copy. Use `ggml_backend_tensor_copy_async`
      (ggml/src/ggml-backend.cpp line 511) or a dedicated copy stream to overlap
      the KV transfer with other work, if the copy shows up in profiling.
- [ ] 3.3 Overlap weight streaming with prefill. If the RTX's weight streaming
      from RAM is a bottleneck during prefill, check whether the scheduler can
      prefetch the next layer's weights while the current layer is computing
      (this is existing offload behavior - verify it is active for the RTX
      context).
- [ ] 3.4 Reuse the model weights. If the doubled RAM footprint (two model
      instances) is a problem, investigate whether the two contexts can share the
      same weight tensors (same `llama_model`) while using different per-phase
      device assignments. This is a larger change (per-phase graph/scheduler
      switching in a single context) and is out of scope for the initial
      implementation - only pursue it if RAM is a hard constraint.

### Phase 4 - RTX weight streaming (double buffering) [FUTURE / kiedyś]

Not part of the initial PoC. Revisit once the split (Phase 1-2) works on the
target model. Motivation: the target model (~17 GB, Q4) does not fit in the
RTX's 6 GB, so the RTX streams most weights from RAM during prefill, which is
slow. The idea is to make the RTX pull weights from the shared RAM (the same
physical RAM the iGPU uses via UMA) through a controlled, double-buffered
pipeline instead of the default offload path.

Key insight (pointer reuse): the weights already live in system RAM (loaded
once, shared with the iGPU via UMA). At some point we can keep a pointer to that
RAM and hand it to CUDA, so the RTX reads the weights directly from RAM over
PCIe. This needs an intermediary object that registers the host pointer with the
CUDA context:

- CUDA side: `cudaHostRegister` / mapped memory (`cudaHostAllocMapped`) -
  registers an arbitrary host pointer (the RAM where the weights sit) so the RTX
  knows the physical addresses. The RTX's DMA controller then reads that region
  over PCIe (Gen4 x8, ~15-16 GB/s) without CPU involvement.
- Vulkan side (reference for how the iGPU already does it): UMA means the
  iGPU's "VRAM" is the same physical RAM. Vulkan allocates with
  `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` (on an AMD iGPU that buffer is also
  DEVICE_LOCAL), and `VK_EXT_external_memory_host` lets Vulkan attach directly
  to a system pointer (`void*` from malloc/mmap of the GGUF). No copy.

Two ways the RTX can use the RAM weights:

- Option A - Zero-Copy GEMM: the RTX runs the matmul reading weights live from
  RAM over PCIe. No buffering, but the Tensor cores are limited to the PCIe
  bandwidth (~16 GB/s) instead of VRAM bandwidth.
- Option B - Weight streaming (double buffering): the RTX DMA-fetches a chunk
  of weights from RAM into a small VRAM buffer (e.g. 2 GB), computes it on the
  Tensor cores, and in the background DMA-fetches the next chunk into a second
  buffer. Swap and repeat. (Preferred direction.)

What would need to be built (Option B):

1. A ring of VRAM zones (2x ~2.8 GB for a 6 GB card) - each holds a chunk of
   layers (~19 layers of a 9B Q4_K_M per zone).
2. A per-chunk prefill loop: compute the chunk from zone A while asynchronously
   copying the next chunk from RAM into zone B, then swap.
3. Split the graph by layer chunks with synchronization points - the most
   invasive part: today llama.cpp builds the graph for the whole layer stack at
   once and the scheduler computes it all; inserting per-chunk sync + weight
   buffer swaps would need a new orchestration path.

## 5. Risks and open questions

- Doubled RAM for weights: two model instances hold the weights twice in system
  RAM. Fine for 32 GB with a 7B model, but a concern for large models. Mitigation:
  Phase 3.4 (share weights) or pick a smaller quant for the RTX instance.
- RTX 6 GB headroom: the RTX context holds a few offloaded layers (in VRAM) plus
  the prompt's KV cache (in VRAM, since offload_kqv = true). Verify the prompt
  length + offloaded layers fit in 6 GB; if not, reduce `n_gpu_layers` for the
  RTX or shorten the prompt.
- Cross-device copy cost: the KV copy (RTX VRAM -> RX shared RAM) goes through a
  host staging buffer. For a long prompt this can be a non-trivial transfer.
  Profile it; Phase 3.1/3.2 reduce it if needed.
- Backend non-determinism: CUDA and Vulkan may produce slightly different logits
  for the same input, so the split output may not bit-match the single-context
  baseline. Compare at the token level, not the bit level.
- Sampler state across the boundary: confirm the sampler's internal state (token
  history, etc.) is correctly carried from the prefill phase to the decode phase
  (task 2.3). A mismatch here causes subtle quality degradation.
- SWA / sliding-window models: if the model uses SWA layers, the KV cache layout
  differs per layer (see `is_swa`, `n_swa` in llama-kv-cache.h). The copy must
  handle SWA layers correctly - verify with an SWA model if relevant.

## 6. Key files and symbols (reference)

- src/llama-model.cpp ~1420: per-layer device assignment (`load_tensors`).
- src/llama-kv-cache.cpp ~200: KV cache K/V tensor allocation (offload vs CPU).
- src/llama-kv-cache.cpp 452: `seq_cp` - in-cache cross-stream copy (reference).
- src/llama-kv-cache.cpp 1239: `get_k_storage` - K tensor access per layer.
- src/llama-kv-cache.h 250: `kv_layer` struct (`il`, `k`, `v`, `k_stream`, `v_stream`).
- src/llama-kv-cache.h 330: `layers`, `map_layer_ids`, `v_cells`, `v_heads`.
- src/llama-context.cpp ~1350: `process_ubatch` - graph build/reuse + compute.
- src/llama-context.cpp ~1414: `ubatch.n_tokens > 1` - prefill vs decode distinction.
- src/llama-context.cpp 4030: `llama_memory_seq_cp` - public API wrapper (reference).
- ggml/src/ggml-backend.cpp 488: `ggml_backend_tensor_copy` - cross-device copy.
- ggml/src/ggml-backend.cpp 511: `ggml_backend_tensor_copy_async`.
- common/speculative.cpp ~2508: two-context setup reference (`ctx_other`).
- include/llama.h 541: `llama_init_from_model`; 516: `llama_model_load_from_file`;
  990: `llama_decode`; 582: `llama_get_memory`.
- common/arg.cpp 2412: `--no-kv-offload`; 2811: `--n-gpu-layers`; 2853: `--tensor-split`;
  2880: `--main-gpu`.
