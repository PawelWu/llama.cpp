# Handoff - `--pp-dev` (prefill on a selected device)

Repo: `D:/shared/project/llama.cpp`
Branch: `pp-custom-device` (created from `any-draft-device-fix`, based on upstream master)
Build: `cmd //c build_pp.bat` -> `build/bin/` (MSVC + Ninja)
Old chunked/streaming experiment lives on branch `pp-only-on-selected-device` (commit `a5c92af3a`), see section 10.

Status: the mechanism is implemented and measured on both models. The host-only gate is now **removed** and
the reserve blowup that used to OOM the 6 GB dGPU is fixed (`pipeline_parallel` off when `pp_backend` is set).
The target config `-ngl 99 --device Vulkan1 --pp-dev Vulkan0` now runs with a 177 MiB compute buffer on the
dGPU, i.e. the streaming works and stays far under 6 GB. It is still **slower than the plain baseline on this
machine** (section 8.3). Section 8.4 has the KV-quantized gemma A/B/C/D sweep.

---

## 1. Goal

Run **prompt processing (PP) on one GPU** and **decode on another GPU**:

```
--pp-dev Vulkan0    device that should run the prefill weight ops
--device Vulkan1    device that holds the model / runs decode
```

Motivating model: `D:\shared\ai-models\Qwen3.8-27B-MXFP4.gguf` (15785.56 MiB of weights, 66 layers,
~204 MiB/layer) - does not fit on the 6 GB dGPU, fits on the 32 GB iGPU. Target: prefill on the fast
dGPU, decode on the iGPU, **with all layers offloaded (`-ngl 99`)**.

Hardware:
- `Vulkan0` = NVIDIA RTX 4050 Laptop, 6 GB (~5152 MiB free)
- `Vulkan1` = AMD Radeon 760M, 32 GB UMA (~32009 MiB free) - its "VRAM" is system RAM
- 62 GB system RAM
- this build has **no CUDA backend**, only Vulkan

Test model used for all measurements in sections 5-7: `Qwen3.8-27B-MXFP4.gguf`.
Earlier gemma-4-E4B numbers are kept in section 9 as historical reference only.

---

## 2. What is implemented on this branch

Commit `25515e4c7` - `common : add --pp-dev to pick the prefill device` (38 lines, 3 files):
`common/arg.cpp`, `common/common.h`, `common/common.cpp`. It only orders `params.devices`.

Uncommitted (58 lines, 8 files) - the actual scheduler hook:

| file | change |
|---|---|
| `ggml/include/ggml-backend.h` | new API `ggml_backend_sched_set_offload_backend(sched, backend)` |
| `ggml/src/ggml-backend.cpp` | `int offload_backend_id` in `struct ggml_backend_sched` (default -1) + forced-offload branch in `ggml_backend_sched_backend_id_from_cur()` |
| `include/llama.h` | `ggml_backend_dev_t pp_backend` in `llama_context_params` |
| `src/llama-cparams.h` | `ggml_backend_dev_t pp_backend` |
| `src/llama-context.{h,cpp}` | `sched_set_pp_backend()`, called after both `ggml_backend_sched_new()` sites in `sched_reserve()` |
| `common/common.cpp` | `cparams.pp_backend = params.pp_dev`; pp device appended to `params.devices` |

Uncommitted also: `sched->offload_backend_min_batch` (32) plus `ggml_backend_sched_op_batch_size()` in
`ggml/src/ggml-backend.cpp`, so the forced branch only captures prefill-sized batches (see 8.8); and an
env-gated `GGML_SCHED_PROFILE` timer in `ggml_backend_sched_compute_splits()` that prints
copy vs compute time per graph.

Plus `src/llama-context.cpp`: `pipeline_parallel` is disabled when `cparams.pp_backend` is set. Without that,
`-ngl 99 --pp-dev` runs with `n_copies = GGML_SCHED_MAX_COPIES = 4`, every cross-backend weight copy is added
as an OUTPUT leaf that is never freed, and `sched_reserve` asks for 4 x model size on the pp device (58-66 GB,
instant OOM). With `n_copies = 1` the copies are graph nodes and ggml-alloc reuses one scratch region per split.

Current state of the forced branch (`ggml/src/ggml-backend.cpp`, ~line 968), **no host-only gate**:

```cpp
if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
    int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
    // the caller selected a backend to run weight ops on, the weights are copied to it
    // the copies are allocated per split and reused by ggml-alloc, so the compute buffer stays small
    if (sched->op_offload && sched->offload_backend_id >= 0 && sched->offload_backend_id != src_backend_id) {
        ggml_backend_t backend = sched->backends[sched->offload_backend_id];
        if (ggml_backend_supports_op(backend, tensor) && ggml_backend_offload_op(backend, tensor)) {
            SET_CAUSE(tensor, "1.off_forced");
            return sched->offload_backend_id;
        }
    }
    ... existing host-offload rule (cause "1.off") ...
    SET_CAUSE(tensor, "1.wgt%d", i);
    return src_backend_id;
}
```

`ggml_backend_offload_op()` is the existing batch-size gate (Vulkan: `ggml_vk_get_op_batch_size(op) >=
op_offload_min_batch_size`, default 32, env `GGML_OP_OFFLOAD_MIN_BATCH`). So **decode (`n_tokens = 1`) is
never forced**, only prefill ubatches.

`common/common.cpp` warns that offloaded weights are copied per prefill batch:
`--pp-dev copies the weights of offloaded layers to the prefill device on every prefill batch`.

**Extra debug hunk that must be reverted before committing**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp:19336`
has `mmap_support = true` instead of `!ctx->is_integrated_gpu` (section 8.2), and `buffer_from_host_ptr` was
tried and reverted.

### 2.1 Debug hunks that MUST be reverted before committing

`ggml/src/ggml-backend.cpp`:
1. `#if 1 // DEBUG: temporary, revert before committing` at ~line 911 (upstream has `#if 0`) - this is what
   makes the assignment dump work.
2. `print_assignments`: two format strings widened from `[%5.5s %8.8s]` to `[%-12s %-14s]` so that
   `Vulkan0` / `1.off_forced` are not truncated.

The `SET_CAUSE(tensor, "1.off_forced")` line is part of the feature and stays.

---

## 3. How the existing mechanism works (CORRECTED)

With `-ngl 0` the weights are placed in a **host** buffer whose buffer type is
`ggml_backend_dev_host_buffer_type(model.devices[0])` - for `--device Vulkan1` that shows up as `Vulkan_Host`.
The scheduler maps that buffer to the **CPU** backend id, so at prefill the existing `1.off` rule moves the
`MUL_MAT`s to the first device backend that wants them.

**Correction to the previous version of this document: no weight copies are made.** Verified on the 15.8 GB
model by listing every `4.cpy` tensor created by `ggml_backend_sched_alloc_graph()`:

- case `-ngl 0 --device Vulkan1`: 9507 `4.cpy` nodes, **0** of them have a weight as source. They are
  activations and KV state (`CPU#conv_state_last-*`, `CPU#new_state-N`, `Vulkan1#norm-N`).
- case `-ngl 0 --device Vulkan1 --pp-dev Vulkan0`: 66 unique copy tensors, all `Vulkan0#norm-N#0` (the
  per-layer RMS-norm output that crosses a split boundary) plus 2 leafs. **No weight copies.**

**Second correction (see 8.7): the "zero copy" claim below is NOT supported by the code.**
`ggml_backend_vk_device_supports_buft()` rejects `Vulkan_Host`, so no Vulkan backend owns those buffers.

Vulkan accepts the host buffer type (`ggml_backend_sched_buffer_supported()` is true), so the GPU
dereferences the pinned host allocation **in place, zero copy**. That is the whole reason `-ngl 0` prefill
runs at 45-167 t/s instead of CPU speed, and it is why the compute buffer stays small
(Vulkan1 535 MiB, Vulkan0 535 MiB) while 15.8 GB of weights sit in RAM.

Consequence: **there is no per-layer weight streaming subsystem and none is needed** as long as the weights
are reachable from the device that computes. The open problem is purely *reachability* of weights that live
in another device's buffer.

---

## 4. Measurement methodology (IMPORTANT)

Use `llama-cli`, not the server. Exact config used for every number in sections 5-7:

```sh
GGML_SCHED_DEBUG=2 ./llama-cli.exe -m "$MODEL" -c 85248 -b 1024 -ub 1024 -fa on -ctk q4_1 -ctv q4_1 \
    -t 6 -tb 6 --temp 0 -st -n 4 --offline -f prompt.txt -lv 5 <device args>
# prompt.txt = 1563 tokens, prints: [ Prompt: X t/s | Generation: Y t/s ]
```

Why not the server: earlier server numbers (`--no-cache-prompt`, port 8091) were inflated - the server does a
warmup batch at init and the unified KV cache reuses a common prefix across slots, so a repeated request
measured only a few freshly evaluated tokens. Server said 829 t/s for a config that `llama-cli` measures at
190 t/s. `llama-cli` is reproducible to ~2 %.

Argument gotchas found the hard way:
- `-ngl 0` not `--ngl 0` (`--ngl` -> `error: invalid argument: --ngl`, which silently killed a whole test case)
- `--device CPU` is **not valid** -> `error: while handling argument "--device": invalid device: CPU`.
  CPU is always the implicit last backend; to put weights in RAM use `-ngl N < n_layer`, not `-ts` against CPU.
- draft layers are `-ngld`, not `-ngd`.

### 4.1 Assignment trace (works now)

The `SET_CAUSE` / `print_node` block in `ggml/src/ggml-backend.cpp` is upstream `#if 0`. Flip it to `#if 1`
(section 2.1), rebuild, then `GGML_SCHED_DEBUG=2` + `-lv 5` dumps every node with its backend and the reason
(`1.dst`, `1.vsrc`, `1.inp`, `1.off`, `1.off_forced`, `1.wgtN`, `2.sup`, `3.*`, `4.cpy`, `usr`).

`llama_context::sched_reserve()` dumps the pp graph (bs=1024) and the decode graph (bs=1) separately. To read
the decode graph, split the log on `## SPLIT #0:` and take the block with the small split count.

Scripts:
- `test_pp_trace.sh` -> `test_pp_trace.{A,B,C,D}.log` + `test_pp_trace.summary.txt`
- `test_pp_dev.sh` -> `test_pp_dev.{E,F,G}.log` + `test_pp_dev.summary.txt`
- `test_pp_cli.sh` (5-case perf sweep, `-ngl 0` case fixed)

### 4.2 Which model to run, and the live server

`llama-server.exe` normally serves the daily model on this machine (routing mode, it reloads itself if killed).

- **Light tests** (scheduler behaviour, split counts, buffer sizes, gate on/off A/B): use
  `gemma-4-E4B-it-qat-UD-Q2_K_XL.gguf` (3.2 GB, 42 layers) and **do not touch the server**. The small model
  fits next to it.
- **Final / representative numbers**: use `Qwen3.8-27B-MXFP4.gguf` (15.8 GB). This needs the dGPU free, so
  `taskkill //F //IM llama-server.exe //T` first. The server comes back in routing mode, so this is harmless,
  but it is only justified when the big model is actually required.

Killing the server to run the small model is a mistake - it costs the user their serving session for nothing.

---

## 5. Trace results on Qwen3.8-27B-MXFP4 (`test_pp_trace.sh`)

| id | config | `1.off_forced` | `1.off` | splits pp / tg | Prompt t/s | Gen t/s |
|---|---|---|---|---|---|---|
| A | `-ngl 99 --pp-dev Vulkan0` | **0** | **0** | 3 | 43.8 | 4.0 |
| C | `-ngl 99` (control) | 0 | 0 | 2 | 39.3 | 3.6 |
| B | `-ngl 0 --pp-dev Vulkan0` | **11806, all -> Vulkan0** | 0 | 1028 / 97 | **166.8** | 2.5 |
| D | `-ngl 0` (control) | 0 | 11806, all -> Vulkan1 | 1028 / 97 | 45.0 | 2.1 |

Forced-op breakdown in case B: `3409 MUL_MAT, 2642 MUL, 882 ADD, 441 SWIGLU, 329 SOFTPLUS, 322 SSM_CONV,
322 SILU, 256 ROPE, 7 GET_ROWS`. Every assignment is `[Vulkan0 1.off_forced]`, none leaked to Vulkan1.

Readings:
1. **The forced branch works and targets the correct GPU.** 11806/11806 on Vulkan0. Prefill 45.0 -> 166.8 t/s
   (3.7x) purely from redirecting host-weight ops from the iGPU to the dGPU.
2. **It never fires at `-ngl 99`.** See section 6.
3. **`--pp-dev` at `-ngl 99` is a layer-split side effect, not prefill offload.** Because
   `common/common.cpp` appends the pp device to `params.devices`, the free-memory proportional split moved
   `2744.82 MiB` (~13 layers) of real weights from Vulkan1 to Vulkan0, KV went to both GPUs, splits 2 -> 3,
   prefill 39.3 -> 43.8. That is a different feature pretending to be this one and should be decoupled.
4. Decode is untouched by design (batch gate).

---

## 6. Why `--pp-dev` did nothing at `-ngl 99` (historical, gate is now removed)

Two independent reasons, both verified at the time. Reason 2 is the one that section 8 removed.

2 is fixed by deleting `ggml_backend_buffer_is_host()` from the forced branch, but reason 1 still stands:
`GET_ROWS` on the host embedding is never offloaded (`ggml_vk_get_op_batch_size(GGML_OP_GET_ROWS) == 0`),
and the reserve blowup that reason 2 was protecting against is now handled by disabling `pipeline_parallel`
when `pp_backend` is set.

1. **There is nothing to offload.** `src/llama-model.cpp` pins the input embedding to CPU
   (`pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list }`), so the only host-resident weight is
   `token_embd.weight` = 1288.28 MiB. It is consumed by `GET_ROWS`, and
   `ggml_vk_get_op_batch_size(GGML_OP_GET_ROWS)` returns 0, so `0 >= 32` fails and **both** the forced
   branch and the standard `1.off` rule decline. Trace confirms: cases A and C have zero `1.off_forced` and
   zero `1.off`; the only causes present are `1.dst`, `1.wgtN`, `2.sup`, `usr`.
2. **Device-resident weights are excluded by design.** The forced branch requires
   `ggml_backend_buffer_is_host(src->buffer)`. Weights in Vulkan1's device buffer are not host memory, and
   `ggml_backend_supports_buft(Vulkan0, Vulkan1_buft)` is false, so Vulkan0 cannot read them in place.

---

## 7. Host weights vs device weights: experiments (`test_pp_dev.sh`)

| id | config | weights | splits pp / tg | Prompt t/s | Gen t/s |
|---|---|---|---|---|---|
| E | `--device Vulkan1 -ngl 50` | 11195 MiB Vulkan1 + 4590 MiB host | 258 / 26 | 41.2 | **2.68** |
| F | E + `GGML_OP_OFFLOAD_MIN_BATCH=1` | same | 258 / 296 | 40.2 | 1.31 |
| G | `--device Vulkan1 -ngl 0` + `MIN_BATCH=1` | 15785 MiB host | 1028 / 1186 | 43.3 | 0.48 |
| D | `--device Vulkan1 -ngl 0` | 15785 MiB host | 1028 / 97 | 45.0 | 2.1 |
| C | `--device Vulkan1 -ngl 99` | 15785 MiB Vulkan1 | - | 39.3 | 3.6 |

Answers to the questions that were open:

1. **Does PP still happen on Vulkan1 when some weights are on CPU?** Yes, already, today, no code needed.
   Case E: prefill graph 258 splits (the 16 host-resident layers are offloaded to Vulkan1 by the existing
   `1.off` rule), decode graph 26 splits with `4619 Vulkan1 / 1580 CPU / 404 usr` nodes - the host-resident
   layers stay on CPU at bs=1. Case D is the same at the extreme: 864 `MUL_MAT` offloaded at bs=1024,
   **865/865 `MUL_MAT` on CPU at bs=1**.
2. **Can Vulkan1 use the same weights in RAM that the CPU backend uses?** Yes, zero copy (section 3).
3. **Can Vulkan1 decode from those host weights?** Mechanically yes - the only thing stopping it is the
   batch gate. Set `GGML_OP_OFFLOAD_MIN_BATCH=1` and the decode matmuls do move to Vulkan1 (case G: 28851
   `1.off` at bs=1, 1186 splits). Practically it is a disaster: **2.1 -> 0.48 t/s**, and E -> F is
   2.68 -> 1.31 t/s. At bs=1 the GPU pulls every weight across the bus to multiply one vector, nothing to
   amortize, plus 1186 CPU/GPU splits of ping-pong. **The gate is a perf heuristic that works; do not
   remove it.**

Best-so-far per axis: prefill = case B (166.8 t/s, host weights + pp-dev on dGPU). Decode = case C
(3.6-4.0 t/s, all weights in iGPU memory). No config today achieves both, which is the whole point of
section 8.

---

## 8. Experiment: drop the host-only gate (done on the small model)

Goal restated: `-ngl 99` (all weights in iGPU memory) **and** prefill executed by the dGPU.

Hypothesis: the comment "copying device weights would need a compute buffer as large as the model" is too
pessimistic, because with `n_copies == 1` the cross-backend copies are added as graph **nodes**
(`ggml-backend.cpp:1515-1524`) with a lifetime of `[dep node, last consumer]`, so ggml-alloc reuses one
scratch region layer by layer.

**The hypothesis is confirmed mechanically, but the result is slower on this machine.**

A/B on `gemma-4-E4B` (3.2 GB, 42 layers), config `--device Vulkan1 -ngl 42 -ts 100,0 --pp-dev Vulkan0`
(all 42 layers forced into Vulkan1's device buffer, so the pp device holds **no** weights and every
offloaded matmul needs a copy). Logs `test_gate_off.I.log`, `test_gate_on.I.log`.

| build | `-ub` | Prompt t/s | Gen t/s | splits (bs=2048) | Vulkan0 compute | Vulkan1 compute |
|---|---|---|---|---|---|---|
| gate ON (committed) | 2048 | 252.7 / 266.5 / 280.4 | 12.3 - 14.1 | **24** | 409 MiB | 388 MiB |
| gate OFF | 2048 | 185.8 / 189.2 / 192.0 | 15.0 - 17.6 | **695** | 443 MiB | 114 MiB |
| gate OFF | 4096 | 185.4 | 17.4 | 695 | 990 MiB | 240 MiB |

Prefill graph node assignment, same config:

| build | MUL_MAT assignment |
|---|---|
| gate ON | `503 [Vulkan1 1.wgt]`, `15 [Vulkan0 1.off_forced]` (only the host weights) |
| gate OFF | `517 [Vulkan0 1.off_forced]` with `Vulkan0#blk.N.attn_q` as `4.cpy` sources, Vulkan1 keeps only `46 SET_ROWS` + `23 FLASH_ATTN` |

Findings:
1. **The compute buffer stays bounded.** 443 MiB at `-ub 2048` for a 3.2 GB model, 990 MiB at `-ub 4096`.
   It scales with ubatch size, not with model size. The "buffer as large as the model" comment is wrong.
2. **Device weight copies do happen and work.** `Vulkan0#blk.0.attn_q (2M) [NULL 4.cpy]` feeds the forced
   `MUL_MAT`. Cross-device Vulkan copy goes through `ggml_vk_buffer_copy()` `MULTI_DEVICE`
   (`ggml-vulkan.cpp:9126-9135`).
3. **The cost is split explosion: 24 -> 695 splits.** Every weight copy is a split boundary, so the ubatch
   is chopped into ~700 CPU/GPU handoffs. Prefill got **25-30 % slower** (266 -> 190 t/s).
4. `-ub 4096` does **not** amortize it here: splits stay 695, buffer doubles, t/s unchanged.
5. Decode assignment is byte-identical between the two builds (`503 [Vulkan1 1.wgt]` + `15 [CPU 1.wgt]`,
   5 splits), so the Gen difference (12.3-14.1 vs 15.0-17.6) is not explained by the scheduler - treat it
   as noise/contention from the `llama-server.exe` that holds 5.7 GB of the dGPU.

Caveat: Vulkan0 had only **193-211 MiB free** during all of these runs (`nvidia-smi`), the AI-serving
`llama-server.exe` holds 5.7 GB of the 6 GB dGPU. So the perf comparison is not on clean hardware.

**Current source state: the gate is back ON** (reviewed state). To rerun the experiment, delete
`ggml_backend_buffer_is_host(src->buffer)` from the forced branch at `ggml-backend.cpp:973`, rebuild
(`cmd //c build_pp.bat`, ~1 min, only ggml-base recompiles).

### 8.1 Big model, gate removed (`big_rev1.log`, `big_rev2.log`, `big_ctrl.log`)

`Qwen3.8-27B-MXFP4.gguf`, `-c 8192 -b 1024 -ub 512 -fa on -n 64 --temp 0`, 3412-token prompt, server killed
first. The reserve blowup is gone: Vulkan0 compute buffer is **177.04 MiB** in both pp-dev runs (was a 58-66 GB
reserve request with `pipeline_parallel` on).

| config | weights | Vulkan0 KV / compute | splits bs=512 / bs=1 | Prompt t/s | Gen t/s |
|---|---|---|---|---|---|
| `-dev Vulkan1 -ngl 99` (baseline) | Vulkan1 14497 MiB | - | 2 | **52.4** | **4.05** |
| `-dev Vulkan1,Vulkan0 -ngl 99` (control) | V0 2744.8 + V1 11752.5 + host 1288.3 MiB | 64 / 177 MiB | - | - | - |
| `-dev Vulkan1 -ngl 99 --pp-dev Vulkan0` | V0 2744.8 + V1 11752.5 + host 1288.3 MiB | 64 / 177 MiB | 1042 / 89 | 64.6 | 2.47 |
| same + `-ot ".*=Vulkan1"` (all weights on iGPU) | Vulkan1 15785.6 MiB, **Vulkan0 holds no weights** | 64 / 177 MiB | 1140 / 153 | 26.8 | 2.43 |

Readings (unquantized KV, superseded by 8.1b):
1. The mechanism is exactly what was asked for: with `-ot ".*=Vulkan1"` the dGPU holds **zero** model weights
   and still executes every prefill matmul, streaming them out of the iGPU buffer, 177 MiB of compute buffer.
2. With f16 KV the dGPU run was slow (26.8 t/s), see 8.1b for why.

### 8.1b Big model with `q4_1` KV, copy/compute attribution (`bigq_{a,d,dp,cp}.log`)

Same configs, `-ctk q4_1 -ctv q4_1` added. `GGML_SCHED_PROFILE=1` (temporary hunk in
`ggml_backend_sched_compute_splits`, syncs after every copy and every split compute, prints per graph compute)
gives the copy vs compute split.

| run | config | Prompt t/s | Gen t/s |
|---|---|---|---|
| A | `-dev Vulkan1 -ngl 99` | 58.8 | **3.75** |
| D | `--pp-dev Vulkan0 -ot ".*=Vulkan1"` | **73.5** | 2.98 |
| Dp | D + profile | 73.7 | 2.68 |
| Cp | `--pp-dev Vulkan0` (weights split) | 73.4 | 2.15 |

Per 512-token ubatch in D (profiled, serialized):
- copy: 13.2 GB of weights in 2.0-3.3 s (avg ~2.9 s, ~4.5 GB/s effective Vulkan1 -> Vulkan0)
- compute: 0.6-3.3 s, growing as the KV cache fills
- copy is ~55% of prefill time, compute ~45%
- Dp == D in total time, i.e. the async copy pipeline is already effectively serialized, no overlap gained

Findings:
1. **With `q4_1` KV, `--pp-dev Vulkan0` beats the baseline on prefill: 73.5 vs 58.8 t/s (+25%).** The RTX 4050
   is faster than the 760M iGPU and the streaming cost is amortized over the 512-token ubatch.
2. The f16 KV run (26.8 t/s) was attention-bound on the dGPU; `q4_1` KV is what makes the streaming path
   competitive. It is not optional for this feature.
3. C == D on prefill (73.4 vs 73.5): moving 2.7 GB of weights onto the dGPU saves nothing, costs decode
   (2.15 vs 2.98). `-ot ".*=Vulkan1"` is the right variant.
4. Decode: no weight copies at bs=1 (the batch gate keeps them on the iGPU), ~270 small activation copies per
   token. Decode is slower than baseline in every pp-dev variant.

### 8.2 mmap / mlock / zero-copy import (all negative)

- `mmap_support = true` for integrated GPUs (`ggml-vulkan.cpp:19336`, still in the tree as a debug hunk):
  big model `-ngl 0` Gen 2.76 -> 2.73 t/s on gemma, 1.88 t/s on the big model. No effect.
- `GGML_OP_OFFLOAD_MIN_BATCH=1` (decode matmuls forced onto the GPU with host weights): gemma Gen 2.76 -> 2.76,
  big model Gen 1.88 -> **0.21** t/s, 1186 splits at bs=1. Confirms the batch gate is correct.
- `buffer_from_host_ptr` (`VK_EXT_external_memory_host`, the real zero-copy import): the capability was flipped
  on and llama.cpp did call it (`src/llama-model.cpp:1760`), the AMD driver rejected every import with
  `ErrorInvalidExternalHandle`, and separately `max_buffer_size` on Vulkan1 is under 3 GB so even a small import
  path would have to chunk. All related edits were reverted; only the `mmap_support` hunk remains.

### 8.3 Why the streaming path cannot win here

Prefill on the iGPU already reads the weights from the same physical RAM the iGPU buffer lives in
(32 GB UMA). Streaming them over to the dGPU means the 6 GB card pays the copy *and* the compute, while the
card that could have computed for free sits idle. The copy volume per ubatch scales with model size, the
compute only with ubatch size, so the bigger the model the worse the ratio.

### 8.4 gemma-4-E4B sweep with `q4_1` KV (`kv_{a,b,c,d}.log`)

`-c 8192 -b 1024 -ub 512 -fa on -ctk q4_1 -ctv q4_1 -n 64 --temp 0`, 3376-token prompt, run while the
serving server was killed (should have used the small model without killing it, see 4.2):

| id | config | weights | Vulkan0 compute | splits bs=512 / bs=1 | Prompt t/s | Gen t/s |
|---|---|---|---|---|---|---|
| A | `-dev Vulkan1 -ngl 99` | V1 2362 + host 858 MiB | - | 2 | **301.8** | **22.5** |
| B | `-dev Vulkan1,Vulkan0 -ngl 99` | V0 367 + V1 1995 + host 858 MiB | 194.6 MiB | 3 | 230.4 | 19.1 |
| C | B + `--pp-dev Vulkan0` | same as B | 120.3 MiB | 848 / 5 | 203.2 | 14.8 |
| D | `-dev Vulkan1 --pp-dev Vulkan0 -ot ".*=Vulkan1"` | Vulkan1 3220 MiB only | 120.3 MiB | 906 / 43 | 186.1 | 14.4 |

A -> B is the cost of merely having the 6 GB card in the device list (weights and KV get split onto it):
301.8 -> 230.4 t/s prefill, 22.5 -> 19.1 t/s decode. B -> C is `--pp-dev` itself: 230.4 -> 203.2. D shows the
same again with the weights fully on the iGPU. Nothing in this family beats A.

### 8.5 KV cache quantization

`-ctk q4_1 -ctv q4_1` is the user's default and every comparison above must use it. It shrinks KV by ~4x
(gemma at `-c 8192`: 40 MiB + 12.5 MiB draft instead of ~160 MiB) and it changes the fused attention path,
so absolute t/s are not comparable with unquantized runs. It does **not** change the conclusion: the weight
copy volume per ubatch is independent of KV precision, and it is what kills the streaming path. The big-model
rows in 8.1 were measured without KV quantization and are internally consistent (same flags in all three
runs), but they are not directly comparable to the 8.4 rows.

Note: the copy tensor names in the trace dump are cut at 12 characters by `%-12s`, so
`Vulkan0#blk.0.attn_q#0` prints as `Vulkan0#blk.0.attn_q`. Grepping for `#<digits>` misses all weight
copies - this cost a wrong conclusion once already.

### 8.6 Copy microbenchmark (`bench_copy.cpp`, build with `build_bench.bat`)

Synthetic test of the reverse-route pattern: two 512 MiB f16 tensors copied Vulkan1 -> Vulkan0, each
followed by 4 chained 16384x16384x512 f16 matmuls on Vulkan0 (~322 ms per graph). Run while the serving
servers were up, so absolute numbers are inflated; the pattern is stable across 5 runs.

Cross-device copies are 3-stage (`ggml_vk_buffer_copy` multi-device path, `ggml-vulkan.cpp:9113`):
V1 GPU copy -> V1 host-visible staging, CPU memcpy staging -> V0 staging, V0 transfer-queue copy -> dst.

| test | pattern | result |
|---|---|---|
| T1 | copy, sync, repeat | ~48 ms/copy (11 GB/s) |
| T2 | two copies in flight, one sync | same as T1, batching copies alone gains nothing |
| T4c | compute only | 322 ms/graph |
| T4a | copy A, compute A, copy B, compute B | 1302 ms/round; each copy degrades to ~323 ms |
| T4b | copy A, copy B, compute A, compute B | 967 ms/round (-26% vs T4a) |
| T4e | compute, then copy to a buffer compute did NOT touch | 88 ms copy |
| T4d | issue next copy while current compute runs | 725-837 ms/round (-37..-44% vs T4a) |

Findings:

1. The copy itself is not the problem: copying into a buffer the GPU just used as a matmul weight costs
   ~323 ms, copying into an untouched buffer costs ~88 ms. In real prefill every ubatch re-copies weights
   into the just-used slots, which matches the 4.5 GB/s effective rate measured in 8.1b (vs 11 GB/s idle).
2. Answer to "copy 2 tensors before processing": yes, T4b beats T4a (967 vs 1302 ms). Pipelining the copy
   into the compute window (T4d) beats both.
3. Practical implication for the pp-dev path: the scheduler currently syncs after each weight copy. Issuing
   the copy for the next split before computing with the current one (async copy, sync only where a split
   needs its tensor) would hide most of the copy behind compute, at the cost of one extra weight slot live
   in the 6 GB card at a time (a few hundred MB, acceptable).

### 8.7 What `Vulkan_Host` really is, and why decode cannot use it for free

Code facts, verified in this tree:

- `ggml_backend_vk_host_buffer_type_alloc_buffer` (`ggml-vulkan.cpp:17154`) allocates **pinned host
  memory** (`ggml_vk_host_malloc`) and wraps it as a CPU buffer:
  `ggml_backend_cpu_buffer_from_ptr(ptr, size)`, then `buffer->buft = buft`. `is_host` is the CPU one.
- It is one **global** buffer type bound to Vulkan device 0
  (`.device = ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0)`, `.context = nullptr`), not per-device.
  On this machine Vulkan device 0 is the NVIDIA card. The existing comment at `ggml-vulkan.cpp:17188`
  ("Should be changed to return device-specific host buffer type") flags exactly this.
- `ggml_backend_vk_device_supports_buft` (`ggml-vulkan.cpp:20001`) rejects it:
  `if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) return false;` - the host type has a
  different `get_name`. `ggml_backend_dev_supports_buft` (`ggml-backend.cpp:639`) has no host shortcut.
- Therefore `ggml_backend_sched_buffer_supported(Vulkan1, Vulkan_Host weight)` is **false** and the
  scheduler treats those weights as **CPU-owned**. Vulkan1 can never read them in place.

Consequence, measured in `bigq_d9_base.log` / `bigq_d9_mb1.log` (`run_d9.bat`,
`-dev Vulkan1,Vulkan0 --pp-dev Vulkan0 -ngl 99 -ot ".*=CPU" --load-mode none -c 8192 -b/-ub 1536 -fa on
-ctk q4_1 -ctv q4_1`):

| run | PP t/s | TG t/s | weight copies per decode token | Vulkan1 compute buffer |
|---|---|---|---|---|
| `GGML_OP_OFFLOAD_MIN_BATCH=32` (default) | 111.94 | **1.52** | 0.0 MiB (895 activation copies) | 180 MiB |
| `GGML_OP_OFFLOAD_MIN_BATCH=1` | 106.36 | **0.72** | **14497 MiB** | 1468 MiB |

With `MIN_BATCH=1` decode does move to Vulkan1, but because the weights are not reachable in place it
**streams the whole 14.5 GB model through the compute buffer on every token**. That is worse than leaving
decode on the CPU. The earlier prediction of "PP ~123, TG ~2.7" for this config was wrong.

The only placement that gives zero-copy decode is weights in Vulkan1's own **device** buffer:

| weights in | PP t/s | TG t/s |
|---|---|---|
| Vulkan1 VRAM, no `--pp-dev` (A) | 58.8 | **3.75** |
| Vulkan1 VRAM, `--pp-dev Vulkan0` (D) | 73.5 | **2.97** |
| pinned host RAM, `--pp-dev Vulkan0` (D8/D9) | **123** | 1.52 |

So prefill speed and decode speed are mutually exclusive under the current scheduler rules: VRAM-pinned
weights give the best decode and force the slow 3-stage V1->V0 copy for prefill; host-pinned weights give
the best prefill and leave decode on the CPU.

Untried idea that could give both: let a Vulkan device accept its own host buffer type in
`ggml_backend_vk_device_supports_buft`, so `Vulkan_Host` weights are assigned to Vulkan1 and read in place
from pinned RAM (which on a 32 GB UMA part is its own memory anyway). Two blockers:

1. the host buft's `.context` is `nullptr`, so the existing `buft_ctx->device->idx` comparison would fault;
   it would need `buft->device == dev` instead
2. the host buft is bound to Vulkan device 0 (the NVIDIA card here), so it must become genuinely
   per-device first

This changes buffer ownership semantics for every host-override and mmap config. Open an issue before
implementing.

### 8.8 Threshold separation for the forced `--pp-dev` branch

The forced branch used `ggml_backend_offload_op()`, which shares `GGML_OP_OFFLOAD_MIN_BATCH` with the
normal offload path. Setting that env var to 1 to move decode onto a GPU also dragged decode onto the pp
device (0.65 t/s on Vulkan0, 1604 MiB compute buffer).

Now the forced branch has its own floor: `sched->offload_backend_min_batch` (32, set in
`ggml_backend_sched_new`), checked with a new `ggml_backend_sched_op_batch_size()` helper mirroring
`ggml_vk_get_op_batch_size`. Prefill-sized batches are forced to the pp device; batch 1 is not, so decode
falls through to the normal offload path and picks the first GPU in the device list.

Measured effect (8.7): it removes the 0.65 t/s catastrophe, but by itself does **not** make decode fast,
because of the reachability problem in 8.7.

### 8.9 D-series: prefill tuning sweep (`run_d2..d9.bat`)

All rows: Qwen3.8-27B-MXFP4, `-c 8192 -fa on -ctk q4_1 -ctv q4_1 -n 64/128 --temp 0`, same-day comparisons
only (cross-day drift is ~10%).

| id | weights | ub | PP t/s | TG t/s | note |
|---|---|---|---|---|---|
| D | Vulkan1 VRAM (`-ot ".*=Vulkan1"`) | 512 | 73.8 | 2.97 | 3-stage V1->V0 copy |
| D2 | CPU mmap (`-ot ".*=CPU"`) | 512 | 82.5 | 1.59 | 2-stage copy |
| D3 | Vulkan_Host (`--load-mode none`) | 512 | 88.6 | 1.50 | pinned DMA, no mmap page cache |
| D4 | 17 layers on V0 + Vulkan_Host | 512 | 89.4 | 1.30 | partial residency |
| D4 | 17 layers on V0 + Vulkan_Host | 1024 | 107.5 | 1.62 | |
| D4 | 17 layers on V0 + Vulkan_Host | 1536 | **113.9** | 1.62 | best with residency |
| D4 | 17 layers on V0 + Vulkan_Host | 2048 | 53.7 | 1.62 | VRAM spill (V0 buffer 632 MiB) |
| D4 l8 | same + `COPY_LOOKAHEAD=8` | 512 | 49.3 | 1.58 | lookahead harmful, VRAM spill |
| D4 | 17 layers, `MIN_BATCH=1` | 1024 | 106.5 | **0.65** | decode forced to V0, buffer 1604 MiB |
| D7 | 20 layers on V0 | 1024 | 102.7 | 1.46 | too much residency |
| D8 | all Vulkan_Host, no residency | 1024 | 106.9 | 1.71 | |
| D8 | all Vulkan_Host, no residency | 1536 | 114.6 | 1.71 | matches D4 without any residency |
| D8 | all Vulkan_Host, no residency | 2048 | **123.29** | 1.71 | **best PP overall** |
| D8 | all Vulkan_Host, no residency | 3072 | 122.35 | 1.71 | plateau |
| D9 | same as D8 ub1536, `MIN_BATCH=1` | 1536 | 106.36 | **0.72** | see 8.7 |

Findings:

1. `-ot ".*=CPU"` does not mean plain CPU buffers: `select_weight_buft` picks `Vulkan_Host`, so weights land
   in pinned host memory and the copy to V0 becomes a single-stage DMA (~6 GB/s).
2. `--load-mode none` is required with CPU overrides (mmap page-cache re-reads cost ~7%; the loader warns
   about it).
3. **Weight residency on Vulkan0 is not needed.** D8 with zero residency matches D4 at ub1536 and then keeps
   scaling to ub2048 because the freed VRAM absorbs the larger compute buffer (581 MiB at ub2048, 849 MiB at
   ub3072).
4. ubatch size is the dominant lever: weights are streamed once per ubatch, so doubling ub nearly halves the
   copy volume per token. It collapses only when the V0 footprint passes ~4.5 GB (weights + compute buffer +
   desktop), where compute slows 3-10x.
5. `GGML_SCHED_COPY_LOOKAHEAD` is net negative on this hardware and should be removed before committing
   (see 8.10).
6. `GGML_SCHED_PROFILE` writes to `stderr` via `fprintf`, so `--log-file` does not capture it. Redirect
   `2>>` in the bat.

### 8.10 Lookahead double-buffering PoC (negative, remove before commit)

`GGML_SCHED_COPY_LOOKAHEAD` deferred freeing of copy tensors via `alloc_deps` and prefetched weight copies
`L` splits ahead. Correct (identical output at every L) but never faster:

- RTX 4050 reports `single_queue = true` (`ggml-vulkan.cpp:6880`: same queue family and `queueCount == 1`),
  so `transfer_queue` is aliased to `compute_queue` and `async_use_transfer_queue = false`. Copy and compute
  share one FIFO, so GPU-side overlap is impossible on Vulkan0.
- L=8 neutral, L=24/64 actively worse; with any V0 residency it spills VRAM (D4 l8: 49.3 t/s).

Verdict: drop the lookahead code, keep the gate and the profiling helper.

---

## 9. Open questions / decisions

0. **Weight reachability is now the main blocker** (8.7): prefill wants weights in pinned host RAM, decode
   wants them in Vulkan1 VRAM, and no Vulkan backend accepts `Vulkan_Host`. Best current compromise for a
   prefill-heavy workload is D8 ub2048 (123 t/s PP, 1.71 t/s TG); for a decode-heavy workload, config A
   (58.8 / 3.75).

   `common/common.cpp` appends the pp device to `params.devices`, which feeds the layer split in
   `src/llama-model.cpp:1425-1490` and silently moves ~13 layers (2744 MiB) plus 64 MiB of KV to the pp
   device. That is why `-ngl 99 --pp-dev Vulkan0` looks faster (64.6 t/s) than the honest variant that keeps
   every weight on the iGPU (26.8 t/s). The pp backend should be added to the scheduler backend list only
   (`src/llama-context.cpp:426-448`), not to the model device list. `-ot ".*=Vulkan1"` is the current
   workaround for testing.
2. Flash-attention resolution: `resolve_fused_ops()` (`src/llama-context.cpp:529`) disables FA when the
   forced matmuls drag the fused node to another device. Workaround `-fa on`. Unresolved.
3. Is `ggml_backend_sched_set_offload_backend()` the right public API shape, or should `pp_backend` be
   passed through `ggml_backend_sched_new()`?
4. Should `--pp-dev` warn/refuse when the pp device cannot hold the reserve buffer?
5. Draft device inconsistency: cases 2/3 of `test_pp_cli.sh` log `devices=[Vulkan0]` but draft KV/compute
   buffers land on Vulkan1. Related to the `any-draft-device-fix` branch.
6. Should the forced-branch floor (`offload_backend_min_batch = 32`) be a CLI/env knob, or is a fixed
   constant acceptable given that real ubatches are >= 512?

---

## 10. Old branch: chunked prefill with per-chunk weight streaming

`pp-only-on-selected-device` (commit `a5c92af3a`) implements PP over layer ranges `[i_start, i_end)` with
weights copied into a small buffer per chunk. Five real bugs were found and fixed there (compute-buffer
overwrite, weights never restored, double `sqrt(n_embd)` scaling, **`llama_kv_cache::apply_ubatch()` is not
idempotent** -> alternating all-`-inf` attention mask -> NaN, per-layer inputs projected from the hidden state
instead of the token embedding). Output is still garbled: chunked PP does not reproduce the single-graph
result (`sum100` after 4 layers: 184.87 at `chunk_size=2` vs 216.86 at `chunk_size=4`).
That branch also carries a lot of debug prints that must not be committed. Kept for reference only.

---

## 11. Historical reference: gemma-4-E4B-it (3.2 GB, 42 layers, `-c 4096 -b 2048 -ub 2048`)

| id | config | Prompt t/s | Gen t/s |
|---|---|---|---|
| A | `--device Vulkan1 -ngl 42` | 272 - 296 | 17.6 - 18.7 |
| B | A + `--pp-dev Vulkan0` | 85 - 196 | 7.8 - 10.0 |
| C | `--device Vulkan1 -ngl 0` | 267 - 279 | 12.6 - 13.4 |
| D | C + `--pp-dev Vulkan0` | 189 - 191 | 13.0 - 14.1 |
| E | `--device Vulkan0 -ngl 42` | 221 | 4.7 |
| F | `--device Vulkan0 -ngl 0` | 192 | 13.8 - 14.0 |

Measured on that model: `--pp-dev` changed the Vulkan0 VRAM footprint by ~180-200 MiB and everything was
slower because the AI-serving `llama-server.exe` held ~5.5 GB of the 6 GB dGPU. `-ub` mattered (161 t/s at
`-ub 512`, 190 t/s at `-ub 2048`). These numbers are superseded by sections 5-7.

---

## 12. Key files

- `ggml/src/ggml-backend.cpp` - `ggml_backend_sched_backend_id_from_cur()` (~922-990), copy insertion
  (~1375-1430), copy nodes added to the graph copy (~1505-1530), `ggml_backend_offload_op` gate
- `src/llama-context.cpp` - `sched_reserve()`, `sched_set_pp_backend()`, `resolve_fused_ops()` (~529),
  backend list build (~426-448), `process_ubatch()`
- `src/llama-model.cpp` - weight buffer type selection (`host_buffer_type` of `devices[0]`), layer split
  (~1425-1490), `dev_input` pinned to CPU
- `common/common.cpp` - `common_model_params_to_llama()`, `common_context_params_to_llama()`, `--pp-dev` warning
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` - `ggml_backend_vk_host_buffer_type()`,
  `ggml_vk_get_op_batch_size()`, `op_offload_min_batch_size`, `ggml_vk_buffer_copy()` MULTI_DEVICE path
  (~9113-9136), `ggml_backend_vk_buffer_cpy_tensor()` (~17051)
