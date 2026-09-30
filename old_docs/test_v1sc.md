# CUDA0,Vulkan1 split sweep + the 40 GB CUDA0 incident

Model: `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` (2 parts, 47 GB, 49 layers, no MTP layers)
Script: `test_v1sc_split.bat`  ->  logs + summary in `prompts_perf/sweep_v1sc/`
Build: `build/bin` (CUDA + Vulkan, sm_89), `GGML_CUDA_NO_PINNED=1`

## 1. The 40+ GB of CUDA0 VRAM: root cause

The `l` branch of the sweep (`--lazy-mode on`) requested those buffers, and the numbers are in the log:

| run | args | CUDA0 model buffer size (log) |
|---|---|---|
| c5l | `-dev CUDA0,Vulkan1 -ts 95,5 --lazy-mode on` | **43041.58 MiB** |
| c15l | `-dev CUDA0,Vulkan1 -ts 85,15 --lazy-mode on` | **37726.45 MiB** |

Chain:

1. `-ts 95,5` splits the weights, not the layers-to-device mapping only: the CUDA0 share is 95% of ~44.9 GB of
   weights (~42.6 GB), and lazy mode builds one buffer per device sized to the sum of the tensors assigned to it.
2. Fit-params cannot correct that split, because it refuses to run when the user set `-ts`:
   `common_fit_params: failed to fit params to free device memory: model_params::tensor_split already set by user, abort`
   (same cause as the vulkan sweep, `test_v1s.md` section on `-fitt`).
3. So a 43 GB buffer is asked from a 6 GB card. On Windows/WDDM that request is not rejected up front: the driver
   backs the excess with shared system memory, so the machine reports 40+ GB attributed to CUDA0 (dedicated +
   shared) while the allocation is really thrashing DRAM. That is the "server took 40+ GB of CUDA0 VRAM".
4. It then dies later, in the lazy read, not at alloc time:
   `CUDA error: out of memory ... ggml_backend_cuda_buffer_set_tensor (ggml-cuda.cu:789)` during the
   `Loading model...` / warmup phase.

Both runs died there (`c5l` at 3.58, `c15l` at 5.19 wall clock). No row was appended to `summary.txt` for either
(the bat only appends when it finds a `Prompt: ` line), and the sweep never reached `c5cm`/`c15cm` - their logs
are still the stale 431-byte `invalid device: CUDA0` error files from the 13:01 launch with the wrong BIN. The
detached bat died with the restart. Current state: no llama process alive, CUDA0 back to 656 MiB used.

Same trap in the daily preset: `[Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-C0+V1]` is `device = CUDA0,Vulkan1` +
`lazy-mode = on`. Loading that preset is the same request; it is what to avoid, not a one-off sweep artifact.

Rule: on CUDA0, `--lazy-mode on` combined with a user `-ts` is not usable for any model that does not fit in
6 GB. Either drop `-ts` (let fit decide), or drop lazy mode and use the host-weights path.

## 2. Results that did complete

| tag | dev order | ts | extra | prompt t/s | gen t/s | weights |
|---|---|---|---|---|---|---|
| c0h | CUDA0 | 100 | `-ot .*=CPU` | 31.9 | 1.7 | CPU_Mapped 44850 + 27465 MiB, CUDA0 0 |
| c5h | CUDA0,Vulkan1 | 95,5 | `-ot .*=CPU` | 48.3 | 1.0 | CPU_Mapped, no device buffers |
| c15h | CUDA0,Vulkan1 | 85,15 | `-ot .*=CPU` | 49.1 | 0.8 | CPU_Mapped, no device buffers |
| c0l | CUDA0 | 100 | `--lazy-mode on` | 25.8 | 1.8 | CUDA0 2373 MiB (fit moved experts to host) |
| c5l | CUDA0,Vulkan1 | 95,5 | `--lazy-mode on` | - | - | CUDA0 43041 MiB -> OOM |
| c15l | CUDA0,Vulkan1 | 85,15 | `--lazy-mode on` | - | - | CUDA0 37726 MiB -> OOM |

Vulkan arm for comparison (`prompts_perf/sweep_v1s/summary.txt`):

| tag | dev order | ts | extra | prompt t/s | gen t/s |
|---|---|---|---|---|---|
| s0h | Vulkan0 | 100 | `-ot .*=CPU` | **176.2** | 2.9 |
| s0l | Vulkan0 | 100 | `--lazy-mode on` | **194.2** | 1.5 |
| s5cm | Vulkan0,Vulkan1 | 95,5 | `--lazy-mode on --cpu-moe` | 119.2 | 1.6 |
| s15cm | Vulkan0,Vulkan1 | 85,15 | `--lazy-mode on --cpu-moe` | 118.2 | **4.2** |

Readings:

1. The `h` branch is the only safe multi-device shape: with `-ot .*=CPU` no weight buffer is created on either
   device (`CUDA_Host 0.00`, `CPU_Mapped 44850 MiB`), so nothing can blow up at load, whatever the split.
2. CUDA0 is the wrong device for host-weight prefill on this box: 31.9 t/s against 176.2 on Vulkan0 (5.5x).
   Cause is already diagnosed in `tasks.md` ("CUDA0 as pp device: closed"): `ggml_backend_cuda_cpy_tensor_async`
   needs both sides CUDA, so each host weight costs a backend sync, a malloc, a CPU read and a pageable H2D
   copy. A Vulkan destination uses the vk staging path instead (~4.5-5.5 GB/s).
3. A Vulkan1 share HELPS CUDA0 prefill (31.9 -> 48.3 -> 49.1) and costs generation (1.7 -> 1.0 -> 0.8). Every
   Vulkan1 share in the Vulkan arm does the opposite and costs prefill. Different bottleneck on each side.

## 3. Recommendation for this model: `-ot .*=CPU` yes, `--pp-dev` no

`-ot .*=CPU`: yes, and it is the safe default for this model.

- It is what removes the 43 GB device-buffer request entirely (section 1). With mmap the weights stay in
  file-backed `CPU_Mapped` pages, so 44 GB of RAM is not committed either - worth keeping on a 63 GB machine
  where the `-ot` + `--load-mode none` variant wants ~44 GB of real RAM (only ~40 GB free mid-sweep).
- Measured: `-dev Vulkan0 -ot .*=CPU` = 176.2 / 2.9, the best prefill of any *loadable* config with sane
  generation. `-dev Vulkan0 --lazy-mode on` is faster on prefill (194.2) but halves generation (1.5).

`--pp-dev`: not for this model, in either device pairing.

- `--pp-dev CUDA0` is a closed dead end: the help text says the weights are copied to the pp device on every
  prefill batch, and the CUDA destination copy path (above) is 1.8x slower than the Vulkan one at 15.8 GB
  (`tasks.md`: 74 vs 135 t/s). A 44 GB model makes that gap worse, not better.
- `--pp-dev Vulkan0` with `-dev Vulkan1` (the shape that wins on the 27B dense) needs the weights to be
  readable by V1 in place, i.e. in `Vulkan1_Host`, which means `--load-mode none` and a committed 44 GB that
  does not fit in free RAM. With mmap the weights are in `CPU_Mapped` and both devices re-copy them per batch:
  44 GB per prefill ubatch (21 MB/token at ub 2048), so prefill cannot beat the ~176 t/s that `s0h` already
  measures with the same copy volume, and decode does not get the zero-copy win the 27B gets because the
  weight set is 3x larger.
- `--pp-dev-resident` would need a real layer share on the 6 GB card; 10-15% of 44.9 GB is 4.5-6.7 GB. At the
  27B that trade cost generation (A15r 142.4/5.4 against A20r 72.6/3.1) and the gain here would be bounded to
  the resident slice. Not worth it on a 47 GB model.

Practical answer: this model belongs on Vulkan0 (RTX 4050) with `-ot .*=CPU`, or on Vulkan0,Vulkan1 with
`--cpu-moe` + lazy when generation matters more than prefill. CUDA0 gets no role on it.

## 4. Still open

- `c5cm` / `c15cm` (CUDA0,Vulkan1 + `--lazy-mode on --cpu-moe`) were never run. `--lazy-mode on --cpu-moe` is
  the one combination that loaded the Vulkan1 shares (s5cm/s15cm), so the CUDA pair is the only unmeasured cell
  of the grid. Expected to lose to the Vulkan arm for the copy-path reason in section 2, so low priority.
- If the CUDA+vk split is going to be measured properly later, the `l` rows must drop the explicit `-ts`
  (fit-params then picks the split) or be replaced by `--cpu-moe` rows.
