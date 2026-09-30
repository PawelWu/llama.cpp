# Plan: per-split PP/TG timeline tracing (what waits for what)

Goal: a timeline where every PP ubatch and every TG token is broken into the actual steps, so we can
read off lines like

```
TG  ctx=main  graph 17  bs=1   total 248.1 ms
   split  0  V1  blk0..blk3      compute  11.2 ms   (Vk GPU 9.8 ms)   copy 0 B
   split  1  V0  blk58..blk60    wait 3.4 ms (prev V1 event)  copy 274 MiB in 41.9 ms  compute 6.1 ms
   split  2  V1  blk4..blk57     wait 0.2 ms  copy 0 B  compute 176.4 ms
   ...
   unaccounted (host, between phases) 6.9 ms
```

and then a graph (Perfetto/Chrome trace, or a Gantt) striped by backend, block range and phase.

The point is attribution: copy vs wait vs compute vs host overhead, per backend, per block range, at
both batch sizes. Today we only have totals (`Prompt: x t/s | Generation: y t/s`) and the split count.

Scope note: this is analysis tooling for the private branch. Nothing here needs to be upstreamable,
and nothing should change scheduling behavior when the tracer is off.

## 1. Questions the trace has to answer

1. TG, one token: where do the ~250 ms go? Compute on the weight backend (Vulkan1 reading
   `Vulkan1_Host`), cross-device waits, host gaps between splits, or the MTP draft context?
2. TG: how many bytes are copied per token, and between which backends? (Expected answer today: 0 B,
   because `ggml_backend_offload_op` is false at batch 1. A non-zero number here is a bug, and the
   trace should make it obvious.)
3. PP, one ubatch: how much time is weight copy to the pp device (V0) vs compute on V0 vs
   synchronization? Is the copy at DMA speed (`bench_copy.cpp` models exactly this b1->b0 weight copy)
   or is it doing something silly like a staging buffer per split?
4. What does weight residency actually save? (compare `-ts 100,0` vs `-ts 90,10` on bytes copied and
   on V0 busy time.)
5. Why does PP collapse between `ts 90,10` and `ts 85,15` (`pp_dev_ot_cpu_fix.md` section 2) when the
   compute buffers are identical? Is it longer compute, longer waits, or a slower copy?
6. Which of the ~97 (or 689) splits at bs=1 are pure serialization overhead, i.e. what is the
   extrapolated cost of one GPU->GPU boundary on this machine?

## 2. What the tree already gives us (no code changes)

- `GGML_VK_PERF_LOGGER=1` (`ggml-vulkan.cpp:18789-18819`): per-node GPU timestamps from a Vulkan
  query pool, converted with `limits.timestampPeriod` and printed by `vk_perf_logger::print_timings`
  (`:2364`). Variants: `GGML_VK_PERF_LOGGER_CONCURRENT=1` (groups fused nodes), and
  `GGML_VK_PERF_LOGGER_FREQUENCY=N` (`:7857-7868`). This is the device-side ground truth per Vulkan
  device and needs nothing new. It costs an extra submit + waitForFences per graph (`:18789`), so it
  changes absolute timings, but it is fine for per-node attribution.
- `GGML_SCHED_DEBUG=2`: `ggml_backend_sched_print_assignments` (`ggml-backend.cpp:1030-1060`) prints
  every split, its backend, its input tensors with sizes (`bytes_in`), and per node its cause
  (`SET_CAUSE`, `:914-918`, values like `1.off_forced`, `1.off`, `1.wgt0`, `4.cpy`). That is the
  static structure: which node runs where and why.
- `sched_reserve: graph splits = N (with bs=1024), M (with bs=1)` and `graph nodes = ...`
  (`llama-context.cpp:744-750`): the two batch shapes we care about.
- `-lv 5` gives `draft acceptance`, `slot print_timing`, and the `--pp-dev copies the weights ...`
  warning.
- `bench_copy.cpp` (untracked, in the repo root) microbenchmarks the cross-device weight copy and is
  the reference for "is our copy at DMA speed".
- CUDA has no per-op profiler here (only `cudaEventRecord` for internal copies). CUDA0 is not in the
  current configs, so it can stay out of phase 1 and get a host-timer fallback later.

## 3. Why that is not enough for a timeline

- Vulkan node timestamps have no host correlation: they are device-clock values, so they cannot be
  placed on the same axis as host phases, and they say nothing about who was waiting.
- Host timing of a split is not its duration: `llama_context::graph_compute` (`llama-context.cpp:2538`)
  calls `ggml_backend_sched_graph_compute_async`, so the call returns after enqueue. The real
  completion is only observable at the next `ggml_backend_event_synchronize`/`event_wait` or at the
  final synchronize.
- Nothing counts copied bytes, and nothing records why a wait happened (previous split's event, this
  split's own backend event, or a full `ggml_backend_synchronize` fallback).
- Nothing tags a graph compute with PP vs TG, the ubatch size, or which context (main vs MTP draft)
  it belongs to.

## 4. Trace format

One JSON-lines record per event, appended to a file, flushed at exit or at a `GGML_SCHED_TRACE_FLUSH`
boundary. Two record kinds:

```json
{"k":"graph","call":17,"t_us":123456789,"ctx":"main","batched":1,"n_tokens":1024,"n_splits":1113}
{"k":"split","call":17,"split":0,"backend":"Vulkan1","dev":"Vulkan1","n_nodes":812,
 "first":"blk0","last":"blk3","bytes_in":1048576,"wait_prev_us":0,"wait_dst_us":12,
 "copy_us":0,"bytes_copied":0,"submit_us":34,"sync_kind":"event_sync"}
```

Fields that do the attribution work:

| field | comes from | why |
|---|---|---|
| `t_us` | `ggml_time_us()` | single monotonic host clock, all phases comparable |
| `batched` | `ubatch.n_tokens > 1` | splits the PP and TG populations |
| `backend` / `dev` | `sched->backends[split->backend_id]` | per-device busy/idle |
| `first`/`last` | first/last node tensor names in `split->graph` | block-range attribution (`blk23.*`) |
| `bytes_in` | `ggml_nbytes(split->inputs[i])` | copy candidates, and the MoE expert path actually copies less |
| `bytes_copied` | real argument of the copy call | the honest copy volume |
| `wait_prev_us`, `sync_kind` | the `if (split->n_inputs == 0 && prev_backend_id ...)` block | GPU-to-GPU boundary cost |
| `wait_dst_us` | the `event_wait`/`synchronize` before each input copy | host blocked on the dst backend |
| `copy_us`, `submit_us` | wall time around the call | copy vs enqueue |

## 5. Implementation, in order

### S1 (80% of the value): the host-side split spine

Anchor: `ggml_backend_sched_compute_splits` (`ggml-backend.cpp:1684-1866`). Six hooks, all inside the
existing loop, all gated by one env var (`GGML_SCHED_TRACE=<path>`), no behavior change:

1. top of the split loop: `t0` + split metadata (`backend`, `n_nodes`, names, `bytes_in`).
2. the `event_synchronize`/`ggml_backend_synchronize` at `:1699-1707` -> `wait_prev_us`,
   `sync_kind = "event_sync" | "full_sync"`.
3. the wait before each input at `:1740-1748` -> `wait_dst_us`.
4. the MoE expert copy at `:1758-1830`: record the real span `(first_id..last_id) * expert_size`
   instead of `ggml_nbytes(input)`. Note this path already calls `ggml_backend_synchronize` and
   `ggml_backend_tensor_get_async` for the ids tensor; log that separately, it is host work.
5. around the copy at `:1832` (`cpy_tensor_async` / `tensor_copy`) -> `copy_us`, `bytes_copied`.
6. around `ggml_backend_graph_compute_async` at `:1833` -> `submit_us`.

Keep the record buffer preallocated (`sched->trace_buf`) and never allocate inside the loop.

Acceptance check: for a single graph, `sum(all phase durations) + host gaps` equals the wall time
between the graph begin and end markers within a few percent, and the split count matches
`ggml_backend_sched_get_n_splits`.

### S2: graph boundaries and the PP/TG tag

Anchor: `llama_context::graph_compute` (`llama-context.cpp:2538`). Emit `graph` records before and
after `ggml_backend_sched_graph_compute_async`, with `batched` (already a parameter), the ubatch token
count (pass it in from `process_ubatch`, `llama-context.cpp:1439`), `this` as `ctx` (main vs MTP draft
context are different objects, and they share the devices), and
`ggml_backend_sched_get_n_splits(sched.get())`.

This is what makes the two populations separable: one `graph` with `batched=1` = a prefill ubatch,
one with `batched=0` and `n_tokens=1` = a TG step.

### S3: join the device-side timestamps

The Vulkan perf logger already prints per-node ns. To join it to the trace, the cheapest path is to
have the tracer record node names in split order (`split->graph.nodes[j]->name`) and then match the
logger's `query_nodes` names per split. A cleaner option, if the logger output proves too coarse, is to
add a `GGML_VK_PERF_LOGGER_JSONL=<path>` output next to `print_timings` (`ggml-vulkan.cpp:2364-2410`)
and emit one record per node with `{name, ns}`. Do this only if the host-side spine cannot answer a
question on its own; the first pass does not need it.

### S4: the pp-dev copy accounting

The `--pp-dev` path is just the `1.off_forced` cause in the split assignment, so it is already visible
in the byte counts of S1: at PP, the splits whose input is a `USAGE_WEIGHTS` host buffer carry the
whole layer. Two extra fields make the analysis exact:

- `src_backend`: `ggml_backend_sched_get_tensor_backend(sched, input)` for each weight input, so we can
  bucket "V1_Host -> V0" separately from "CPU -> V0" (that distinction is the whole mmap vs
  `--load-mode none` difference, see `pp_dev_ot_cpu_fix.md`).
- `cause`: reuse `causes[hash_id(node)]` (already computed under `SET_CAUSE`, `:914`) for the first
  node of the split, so every row says why it landed there (`1.off_forced`, `1.off`, `1.wgt0`, `4.cpy`).

### S5 (later, only if needed): CUDA0

No per-op timestamps exist for CUDA. Options, cheapest first: host `submit_us` only (with CUDA graphs
this is meaningless), `cudaEvent` per split, NVTX ranges + Nsight Systems. Not needed while the useful
configs are Vulkan-only.

## 6. Analysis recipes

Each recipe is a filter over the JSONL plus one small computation.

- R1 TG budget per token (answers Q1). Filter `batched=0` for one context, group by `backend`, sum
  `submit_us`, `wait_prev_us`, `wait_dst_us`, `copy_us`; compare against the graph wall time. The
  remainder is host time between phases.
- R2 PP copy vs compute (answers Q3). Filter `batched=1`, sum `bytes_copied` and `copy_us` and
  `submit_us` per backend. Then compare the observed copy rate against `bench_copy.exe <size_mb>` on
  the same machine, same day.
- R3 what the copy volume costs (answers Q4). Same filter, `-ts 100,0` vs `-ts 90,10` vs `-ts 85,15`:
  `bytes_copied` per ubatch, per-block-range breakdown of where the bytes come from, and the V0 busy
  time. This is the number that decides whether residency is worth a flag.
- R4 serialization overhead (answers Q6). Histogram `wait_prev_us + wait_dst_us` per split at bs=1,
  times the number of boundaries. If the median boundary costs 0.5 ms and there are 97 of them, that
  is 48 ms of a 248 ms token, and the fix is fewer boundaries (fewer devices, or residency), not a
  faster kernel.
- R5 the PP cliff (answers Q5). Compare the same trace at `ts 90,10` and `ts 85,15`: does
  `submit_us` on V0 grow (compute slowed by VRAM pressure) or do `wait_dst_us`/`copy_us` grow (the
  copies got slower)? The two have completely different fixes.
- R6 GPU idle fraction. From the Vulkan node timestamps: sum node ns / graph wall = V0 and V1
  utilization. Anything below about 85% on the weight backend at TG is host-bound, and the trace says
  which phase ate the gap.

## 7. Visualization

Python 3.13 is present (`/c/ProgramData/miniforge3/python`) but matplotlib, plotly and pandas are NOT
installed. So the default output has to be zero-dependency:

1. `trace2perfetto.py` (stdlib only): JSONL -> Chrome trace JSON (`{"traceEvents":[{"ph":"X",...}]}`),
   with `pid` = context (main/draft), `tid` = backend, one X event per phase (`wait_prev`, `wait_dst`,
   `copy`, `submit`, plus a `host_gap` event). Open in `chrome://tracing` or ui.perfetto.dev. This is
   the interactive graph, no install needed.
2. `trace_summary.py` (stdlib only): the attribution tables of R1-R6, plus an ASCII waterfall for one
   graph so it works over a terminal:

```
call 17 bs=1 V1 |==blk0-3 9.8ms==|          .             |=====blk4-57 176.4ms=====|
call 17 bs=1 V0 |                |w 3.4|== 41.9ms copy ==|==6.1ms==|
call 17 host   |=enq=|===host gap 4.1ms===|...
```

3. Optional, only if wanted: `pip install plotly` for a Gantt chart per token as an HTML file.

Suggested location: `prompts_perf/trace/` (scratch area, next to the sweeps), not `tools/`.

## 8. Measurement protocol

- Fixed config for every comparison: `--load-mode none -dev Vulkan1,Vulkan0 --pp-dev Vulkan0
  -ot ".*=CPU" -fa on -ctk q4_1 -ctv q4_1 -c 85000 -b 1024 -ub 1024 -n 32 -st`, one fixed prompt.
- Warm up: discard the first PP ubatch and the first 4 TG steps (VRAM/page-cache warming). Compare
  steady state only.
- Always record with and without the tracer on the same config; if the ratio is worse than 2%, report
  both numbers and fix the tracer (preallocated buffer, one write per graph, not per phase).
- Never add a `synchronize` that was not there. Trace what the code does, do not "clean it up".
- Keep the two contexts separate: main and MTP draft are separate scheds on the same devices, so
  concurrent timeline rows are expected and legitimate.
- Device and host clocks are different clocks. Do not try to align them; join by (call, split, node
  name) and present device timings as a per-split overlay, not on the host axis.
- One variable per run. The `-n 32` + MTP protocol makes t/s noisy through draft acceptance
  (0.36-0.78 across runs, see `pp_dev_ot_cpu_fix.md`), so use the trace numbers, not the t/s line.

## 9. Suggested order of work

1. S1 + a minimal `trace2perfetto.py`. Verify the phase sum equals the wall time on one PP graph and
   one TG graph. This alone answers Q1, Q2, Q3, Q6.
2. S2. Without it there is no clean PP/TG split nor per-context attribution.
3. S4 (`src_backend`, `cause`). Needed for anything about `--pp-dev`.
4. R5 (the cliff) and R3 (residency payoff) as the first two real analyses, because they decide
   whether `--pp-dev-resident` should exist at all.
5. S3 (Vulkan per-node join) only if R1/R6 leave the compute phase unexplained.
6. S5 (CUDA) only if CUDA0 comes back into the picture.

## 10. Hypotheses this should settle

- TG is bounded by Vulkan1 reading `Vulkan1_Host` (UMA pinned) at about 30-40 GB/s, so TG lands near
  4 t/s and only VRAM placement (V0 residency, or dropping `--pp-dev` and using the V1 device buft)
  moves it. R1 + R6 say which.
- PP is bounded by the `V1_Host -> V0` weight copy volume divided by the pin-to-device DMA rate, which
  is why `ubatch` dominates and why `--load-mode mmap` (weights in `CPU_Mapped`, an extra hop) is
  slower. R2 + `bench_copy.exe` say which hop is expensive.
- The `ts 85,15` cliff is VRAM pressure on V0 (about 4.5 GB footprint, `handoff.md` 8.9) showing up as
  slower compute, not slower copies. R5 says which.
- Weight residency on the pp device saves only the copy of those layers (a few percent of the prefill
  copy volume), and costs an extra decode boundary. R3 + R4 say whether that is a net win.
