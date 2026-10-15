# The prefill device (`--pp-dev`)

A second GPU (or any other accelerator) can be worth more than a bigger weight split: keep the weights on the
device that holds them, and run only the *prefill-sized* batches on another device whose compute buffer is big
enough. That is what `--pp-dev` selects - one dedicated device for prompt processing.

This page also lists the runtime environment variables that tune the feature. They have no command-line flag, so
they do not appear in `llama-cli --help`.

## Command-line flags

| Flag | Value | Default | Notes |
|---|---|---|---|
| | `--pp-dev` | device name, one only (`Vulkan0`, `CUDA1`, ...) | not set | Weight ops of prefill batches (batch size >= 32) run on this device; the weights are copied into its compute buffer once per graph split. Decode (`n_tokens = 1`) is untouched. A device that is not part of the active device list is ignored with a warning. Env: `LLAMA_ARG_PP_DEV` |
| | `--pp-dev-resident` | - | off | Keep the weights of the layers the layer split assigns to the pp device (`-ts` / `--tensor-split`) inside that device's own buffer type, so they are read in place instead of copied on every prefill batch. Needs `--pp-dev`; if no layer is assigned to it the run warns and keeps nothing resident. Env: `LLAMA_ARG_PP_DEV_RESIDENT` |

## Environment variables

These are read once per process (they are cached), so set them before launching the binary.

| Variable | Type | Default | Effect |
|---|---|---|---|
| `GGML_PP_NORESV_FIX` | value-based: only a leading `1` counts as on | off (`0`) | The scheduler probes decode-sized batches while resolving fused ops, which leaves the pp device switched off; by default it is re-armed for each reserved graph so later prefill batches still land there. Setting this to `1` keeps that switch-off in place - i.e. `1` disables the re-arm, anything else (including `0`) behaves like unset |
| `GGML_PP_NO_NORM` | presence-based: any non-empty value counts as on, `=0` is still **on** | off (`unset`) | RMS norm has no weight source, so it normally follows its input onto the device holding that tensor and gets copied once per consumer on the pp device. By default the op runs on the pp device for prefill-sized batches; setting this variable leaves it with its input instead. Use it to compare split counts on a single binary |
| `GGML_VK_HOST_STAGE` | value-based: any non-zero integer turns it on, `=0` is off | off (`unset`) | Vulkan only. On the multi-device copy path, let the CPU read the mapped host pointer of the source buffer directly instead of staging through a GPU-side copy. Only applies to sources that are host-visible **and** cached (pinned) - uncached UMA memory reads back around 100 MB/s on the CPU. Also frees the source queue, which matters when several devices copy at once |
| `GGML_DRAFT_B` | integer cap | off (`unset`) | Clamp the draft context batch / ubatch size to this value, only for block-diffusion drafts (`--spec-type draft-dflash`, `draft-dspark`). Keeps the flash-attn working set of those drafts small on a dGPU that also stages prefill weights; a draft batch at or below the cap is left alone |
| `GGML_ALLOC_HIGH_WATER` | presence-based: any non-empty value counts as on | off (`unset`) | `ggml/src/ggml-alloc.c`. The allocator re-reserves a graph whenever a tensor no longer fits the `size_max` cached for its slot, and `reserve_n` recomputes those sizes from the current graph only. Graphs whose batch width alternates (speculative drafts: decode width = accepted count, draft width = 1) therefore fail and re-reserve on every growth step. With this variable set, each reserved slot keeps the largest size it has ever needed, so the first wide graph sets the high-water mark and later narrower graphs reuse the plan. Costs memory (the widest layout is kept for good), not speed: measured on `Qwen3.8-27B-UD-Q4_K_XL`, MTP `n_max=3`, `-b/-ub 3072`, the 70-node draft graph went from ~400 reallocs per 512-token run to 5, TG +4 percent; prefill is ~4 percent slower because the padded buffers fit worse |
| `GGML_TRACE_REALLOC` | presence-based: any non-empty value counts as on | off (`unset`) | Same file, companion to `GGML_SCHED_DEBUG_REALLOC` (`ggml/src/ggml-backend.cpp:1990`, which only reports that a same-size realloc happened). Prints one line per tensor that does not fit its cached `size_max` - name, op, type, `ne`, wanted bytes vs cached bytes - so the exact tensor driving the re-reserve is visible. Use it together with `GGML_SCHED_DEBUG_REALLOC=1` to reproduce a draft-oscillation case on one binary |

Note the deliberate difference between the two gates: `GGML_PP_NO_NORM` only checks that the name is set, the
other three parse their value. In `cmd.exe`, `set VAR=` removes the variable rather than making it empty, so use
an explicit value when you want "off".

## Where to read more

- [multi-GPU guide](multi-gpu.md) for the split modes and `-ts` / `--tensor-split` that this feature builds on.
- `prompts_perf/pp_dev_param_sweep.bat` (`prompts_perf/pp_dev_param_sweep.md`) runs every knob above against a
  fixed prompt and checks both placement evidence in the logs and byte-identical generated text.
