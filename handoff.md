# Handoff - `--pp-dev` (prefill on a selected device)

Repo: `D:/shared/project/llama.cpp`
Branch: `pp-host-buft` (created from `any-draft-device-fix`, based on upstream master)
Build: `cmd //c build_pp.bat` -> `build/bin/` (MSVC + Ninja, Vulkan **and** CUDA, sm_89)
Old chunked/streaming experiment lives on branch `pp-only-on-selected-device` (commit `a5c92af3a`), see section 10.
Live session log with the newest numbers: `tasks.md`. This document is the longer-form reference.

Status: **read section 0 first.** The branch moved a long way past what sections 2, 8.7 and 9.0 describe. The
per-device host buffer type is implemented and committed, which removes the weight-reachability blocker that
section 8.7 calls the main problem, and gives Vulkan1_Host weights with zero-copy decode. Both models are
measured. Section 8 is still useful as raw measurement data, but its conclusions about what is impossible no
longer hold.

---

## 0. Current status

The important change since the earlier version of this document is the **per-device host buffer type** in
`ggml-vulkan.cpp`:

- every Vulkan device gets its own `host_buffer_type` (`VulkanN_Host`) next to `buffer_type`, created in
  `ggml_vk_get_device` (~line 7496)
- on UMA devices `ggml_backend_vk_device_get_host_buffer_type` (19411) returns that per-device type; discrete
  GPUs keep the old global pinned `Vulkan_Host`
- the per-device type reuses `ggml_backend_vk_buffer_type_name` as its `get_name`, so
  `ggml_backend_vk_device_supports_buft` accepts it, and `is_host` is NULL so the scheduler treats it as
  device-owned

Consequences, measured on Qwen3.8-27B-MXFP4:

- **8.7 is solved.** A Vulkan device can now claim host-visible weights and read them in place, which is the
  piece 8.7 says cannot work. Weights in `Vulkan1_Host` give the best decode measured so far (TG 4.16 without
  pp-dev, vs 3.75 for the VRAM-pinned baseline).
- **9.0 is partly obsolete.** The pp device is still appended to `params.devices`, but the zero layer-share
  fix in `common/common.cpp` means it no longer claims layers or KV.
- Prefill with `--pp-dev Vulkan0` reaches PP 122-135 at 12k context, flat over the prompt, after the
  FLASH_ATTN_EXT forced-offload fix (tasks.md SEVENTH FINDING).
- The chunked-prefill and "weights are unreachable" analysis below is history. That reachability problem no
  longer exists for integrated GPUs.

Corrections, read these before trusting the older sections:

- Section 2 lists an "Uncommitted (58 lines, 8 files)" scheduler hook. That work is committed, mostly in
  `bf19abeb2`.
- Section 2.1 says the `mmap_support = true` debug hunk at (then) `ggml-vulkan.cpp:19336` must be reverted.
  It already is.
- Section 4.2 says this build has "no CUDA backend, only Vulkan". The current build has both.
- Section 8.10's `GGML_SCHED_COPY_LOOKAHEAD` is still in the tree but env-gated and default off. It is a
  confirmed negative result and must be removed before any commit.

Work sequence on this branch: per-device host buft -> pp-dev + FLASH_ATTN forced offload -> MTP draft on the
MTP dGPU -> CUDA0 as pp device. The last one is closed, see section 13.

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

