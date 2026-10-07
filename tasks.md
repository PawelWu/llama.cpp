# Tasks - per-device host-visible buffer type (`pp-host-buft`)

Branch: `pp-host-buft`, created from `any-draft-device-fix` (b1cb4909c).
Parent work stashed on `pp-custom-device` (stash@{0}, "pp-custom-device WIP: forced offload branch ...").
Note: commit 25515e4c7 (`--pp-dev` arg) is NOT in this branch. It comes back with the stash.

## STATUS (2026-09-30)

Phases 1-3 are DONE and verified. The rest of this file is the chronological session log; findings
are listed here in their final state.

Done:
- per-device host buffer type (VulkanN_Host) for UMA devices, is_host = NULL, all four get_base
  crash paths fixed (Phase 1 + FIRST..FOURTH FINDING)
- pp-dev stash restored and measured on Qwen3.8-27B (FIFTH, SIXTH FINDING; SEVENTH FINDING = the
  FLASH_ATTN forced-offload fix that stopped the long-prompt PP decay: PP 83.8 -> 122, flat)
- CUDA0 as pp device: tested and CLOSED (74.0 vs 132.5-134.9 t/s, copy-path bound, see
  "CUDA0 as pp device: closed")
- MTP-on-dGPU validated end-to-end: TG 6.40-6.63 vs 4.16 no-spec (+59%); backlog item 2 done
- chunked/streaming prefill: ABANDONED (docs moved to old_docs/)
- tree hygiene done: debug prints and A/B toggles removed, junk deleted, closed session docs moved
  to old_docs/ (index in old_docs/README.md)

## STATUS UPDATE (2026-10-02)

Merge `897520425` (any-draft-device-fix -> pp-host-buft) concluded. The custom per-device Vulkan host
buft was adapted to the new `alloc_buffer_n` / `get_alloc_size_n` iface fields (required, the struct
is positional). Review verdict: both conflicts resolved clean, every custom hunk preserved.

Item 1 (TG vs pp-dev) RESOLVED - the old number predated the c6a29d862 TG fix. Measured on the merged
build (GGML_SCHED_DEBUG=1): the decode graph is a single Vulkan1 split (pre-fix h2_scheddump.log
showed the alternating norm-on-Vulkan0 pattern), and TG with/without pp-dev is 4.16 vs 4.22 t/s at
ctx 4096 - within noise.

Item 3 (zero-copy prefill via VK_EXT_external_memory_host) CLOSED - NOT worth it. Probe (bench_import,
untracked): both GPUs expose VK_EXT_external_memory_host and NVIDIA V0 imports a foreign page-aligned
host pointer (OK). But shader reads from imported host memory run at ~6.4 GB/s vs 13.37 GB/s for a
single V0 DMA copy, so in-place compute is 1.65x slower per weight (80.4 ms vs 48.7 ms for 512 MiB,
R=1) and the gap grows with reuse -> staging into VRAM wins. Side finding: the real pp-dev copy path
moves 13.2 GB/ubatch at ~3.1 GB/s (4.22 s) while the same GPU sustains 13.37 GB/s for one DMA, so
the copy path, not weight placement, is the PP lever.

Item 5 (Phase 3 checklist) DONE, and the upstream issue is dropped - this is a private fork.

Open (in order):
1. PP copy path: close the 3.1 -> 13.37 GB/s gap (fewer/larger/async weight copies, or import the
   V1_Host allocation into V0 and DMA from the imported source per ubatch). New, from the 2026-10-02
   B3 investigation.
2. The ~9% MTP decode cost with pp-dev (TG 6.8 -> 6.2). Decode is a single Vulkan1 split, so it is not
   split overhead; needs a per-split profile of the decode graph with MTP on.
3. ub >= 3840 segfault in the QWEN35 graph_reserve / DeepSeek V4 HC probe path. Needs a stack trace
   (WinDbg/cdb or debug build). NOTE: did NOT reproduce at ub 4096/5120 with mxFP4 at -c 32768, so the
   trigger is the context size or the model, not the ubatch alone.

## STATUS UPDATE (2026-10-02, session 2)

Closed: the "lost performance" is not a code regression. Both server logs it came from ran stale
binaries (`d:\progs\llama-cpp-adv` Oct 1 21:02 = before `c6a29d862`; `d:\progs\llama-cpp-pp` Sep 27 =
before the per-device host buft). Controlled A/B on one build is in handoff.md section 11:
host-visible weights cost nothing for decode, pp-dev costs ~9% TG with MTP and ~3% without, and
prefill doubles (54.2 -> 114.1 no-spec, 56.1 -> 111.1 with MTP). Action item is operational:
refresh the deployed copies.

Also done: section 9.0 decoupling (commit `37c345001`) and `--pp-dev-resident` restored
(commit `a2500d093`). Full matrix sweep (ub 2048-5120, ts splits, MTP device, resident, 30k prompt)
in `super_sweep.md`, logs in `prompts_perf/super_sweep/`, harnesses `prompts_perf/super_sweep.bat`
and `prompts_perf/pp_tradeoff.bat`.

Best measured config: `-dev Vulkan1,Vulkan0 -ts 90,10 --pp-dev Vulkan0 -ot ".*=CPU"
--pp-dev-resident` = 118.4 PP / 6.6 TG at 5k; 2.8x ootb prefill on a 30k prompt (116.7 vs 41.8,
and 3.4x on the last chunk). Best prefill overall 135.2 t/s at ub 3072 with MTP.

Context: handoff.md section 0 (corrected), then 8 and 9. Read those first. Closed and superseded
session docs live in old_docs/ (see old_docs/README.md for the index).

## Goal restated

Serve with `-dev Vulkan1 -ngl 99`, weights in pinned host RAM owned by **Vulkan1**, decode runs on
Vulkan1 zero-copy in place, and reuse the existing prefill offload path (`1.off` / the forced branch
from the stash) to run prefill weight ops on Vulkan0. One buffer placement that satisfies both sides
instead of double-buffered weight streaming.

On a 32 GB UMA iGPU the HOST_VISIBLE heap is system RAM - the same physical memory the device buffer
would have used. So "host buffer accepted as Vulkan1's own" should decode at case-A speed (3.75 t/s)
while keeping the d-series prefill path (~123 t/s) available via `--pp-dev`.

## Why the current host type cannot do this (from 8.7, verified in tree)

- `ggml_backend_vk_host_buffer_type_alloc_buffer` (ggml-vulkan.cpp:17154) allocates pinned host
  memory via `ggml_vk_host_malloc` and wraps it with `ggml_backend_cpu_buffer_from_ptr`. There is no
  vk_buffer behind it, so Vulkan shaders cannot bind it.
- It is one global buft bound to Vulkan device 0 (.device = reg dev 0, .context = nullptr).
- `ggml_backend_vk_device_supports_buft` (ggml-vulkan.cpp:20001) rejects it because its `get_name`
  differs from `ggml_backend_vk_buffer_type_name`.
- `ggml_backend_vk_device_get_host_buffer_type` (ggml-vulkan.cpp:19312) ignores `dev` and always
  returns the global one. This is the seam.

## Phase 0 - probe before any scheduler code (do this first)

Extend `bench_copy.cpp` (in the stash only for tracked files; the bench itself was untracked, keep a
copy) with a V1-side probe. Two questions, one benchmark run each:

1. **Compute from host-visible memory on V1.** Allocate a real vk_buffer on Vulkan1 with memory type
   HOST_VISIBLE (prefer HOST_VISIBLE|HOST_CACHED, else HOST_VISIBLE|HOST_COHERENT; pick the heap via
   `vk_instance.devices[1]->vram_heap_idx`-style logic, check what ggml-vulkan already exposes -
   `ggml_vk_create_buffer` / memory-type selection code around the staging allocations). Load a 512M
   f16 tensor, run the 4-chained-matmul graph (same shape as existing T4c). Compare against the same
   matmul from DEVICE_LOCAL. Acceptance: host-visible matmul throughput >= ~70% of device-local.
   On UMA they share physical RAM so it should be close to 100%; if it is far below, stop here and
   record why (uncached host path?). Also measure the fill path: CPU memcpy into mapped memory.
2. **Copy V1 host-visible -> V0.** `ggml_backend_tensor_copy` (or vk_buffer copy) from the mapped
   vk_buffer to a V0 buffer. Expect a 2-stage path (map + memcpy into V0 staging, V0 DMA), no V1 GPU
   copy stage. Acceptance: >= ~6 GB/s (d-series measured ~6 GB/s for the CPU/pinned source path,
   ~4.5 GB/s effective end-to-end in real prefill, ~10-11 GB/s idle DMA).

Decision tree:
- both pass -> Phase 1.
- (1) fails, (2) passes -> weights readable but compute-crippled; the idea is dead for decode.
  Record numbers in handoff and stop.
- (2) fails but (1) passes -> decode still wins; prefill keeps using the D8-style host-CPU copy.
  Implement Phase 1 anyway, prefill is optional.
- both fail -> write up, revert to stash branch, close the idea.

## Phase 1 - per-device host buft in ggml-vulkan.cpp (the only real code change)

Scope: ggml/src/ggml-vulkan/ggml-vulkan.cpp, plus nothing else. llama.cpp needs no change:
`src/llama-model.cpp:1054` and `src/llama-context.cpp:437/2167` call
`ggml_backend_dev_host_buffer_type(dev)`, which routes through the device callback at ggml-vulkan.cpp:20161.

1. Add a per-device host buffer type: one static buft per vk device (device count is small and
   fixed), each with a real context struct `{ vk_device * device }` (mirror
   `ggml_backend_vk_buffer_type_context`). Alloc allocates a vk_buffer on that device with a
   host-visible memory type instead of `ggml_vk_host_malloc` + cpu wrapper.
   - `get_alignment` = `minMemoryMapAlignment` of that device (not device 0's).
   - `get_max_size` = that device's `suballocation_block_size` (or its host-visible heap limit,
     whichever is smaller; verify the AMD host-visible heap reports >= 16 GB, this model needs it).
   - keep `is_host` = true (CPU buft's), that is what makes the loader mmap into it correctly
     (llama-model-loader.cpp:1264 checks exactly this).
   - buffer iface: reuse the standard vk buffer `get_tensor` / `set_tensor` / `cpy_tensor`; they
     already handle mapped host-visible memory because staging buffers use the same path.
   - mapped permanently at alloc time (like staging), unmap+free in free_buffer. Remember the
     mapped pointer in the buffer context for `get_tensor`/`set_tensor` fast paths.
2. `ggml_backend_vk_device_get_host_buffer_type(dev)`: return the buft for `ctx->device->idx`.
   Gate: return the global pinned-CPU one for discrete GPUs (`!ctx->is_integrated_gpu`), same as
   today, because discrete BAR windows are tiny. Integrated GPUs get the new host-visible buft.
3. `ggml_backend_vk_device_supports_buft` needs NO change: the new buft's get_name must be a
   distinct name (e.g. keep `Vulkan_Host` but the name check is `ggml_backend_vk_buffer_type_name`
   for device buffers only). Simplest correct shape: give the per-device host buft its own
   `get_name` and add one branch: `if (get_name == ggml_backend_vk_host_buffer_type_name_device)
   return buft_ctx->device->idx == ctx->device;` before the existing reject. The old global type
   stays rejected, as today.
4. Do NOT touch `mmap_support` (the 19336 hunk stays reverted on this branch) and do NOT touch
   `buffer_from_host_ptr` (8.2 showed the AMD driver rejects VK_EXT_external_memory_host imports;
   a driver-allocated host-visible buffer avoids that path entirely).

Sanity check the whole env before measuring: `-dev Vulkan1 -ngl 0` on the big model must show
`Vulkan1 model buffer size ~15785 MiB` (weights land in the new buft via host_buffer_type of
devices[0]) and `GGML_SCHED_DEBUG=2` decode graph must show weights assigned `[Vulkan1 1.wgt]`
with ~2 splits and 0.0 MiB weight copies per token in `GGML_SCHED_PROFILE=1`.

## Phase 2 - restore the pp-dev stash and measure

1. `git stash pop` (on this branch it applies on top; expect conflicts only if ggml-vulkan.cpp
   touched the same lines - the stash's ggml-vulkan hunk is just `mmap_support`, drop it).
   This restores the forced-branch (`offload_backend_id`, min batch 32), `--pp-dev` plumbing,
   `pipeline_parallel` guard, GGML_SCHED_PROFILE. Per handoff 8.10, re-apply the stash WITHOUT the
   `copy_lookahead` block if convenient, or leave it dormant (env-gated, default off) and remove
   before any commit.
2. Measure on Qwen3.8-27B-MXFP4, methodology of handoff section 4 (`llama-cli`, q4_1 KV, same-day
   comparisons, kill llama-server only for big-model runs):

   | id | config | expect |
   |---|---|---|
   | H1 | `-dev Vulkan1 -ngl 99` | TG ~3.7 (case A parity, zero-copy decode proves the buft) |
   | H2 | H1 + `--pp-dev Vulkan0` | PP >= ~110 (d-series parity, now honest: no device-list side effect), TG ~3.7 |
   | H3 | H2 + `-ot ".*=Vulkan1"` | same as H2 minus the 13-layer side effect (section 9.0 check) |

   Also gemma quick pass (do NOT kill the server): case A/B of 8.4 for regression sanity.
3. Trace check for H2: decode graph identical to H1 (batch gate holds), prefill shows
   `1.off_forced`/`1.off` to Vulkan0 with the host buft now owned by Vulkan1. If the forced branch
   (from the stash) refuses to fire because `src_backend_id != CPU`, that is CORRECT and desired:
   the plain `1.off` path with a higher-priority GPU may already cover prefill; if not, relax the
   forced branch condition to `offload_backend_id != src_backend_id` regardless of host-ness
   (it already is shaped that way in the stashed version).

## Phase 3 - cleanup and tree hygiene (before any commit)

1. Revert debug hunks from the stash: `#if 1` sched dump (ggml-backend.cpp ~911), widened
   `%-12s/%-14s` format strings, GGML_SCHED_PROFILE block, copy_lookahead block (8.10 verdict:
   negative on this hardware, single_queue on the 4050).
2. `--pp-dev` help text in common/arg.cpp: remove the `GGML_OP_OFFLOAD_MIN_BATCH=1` decode advice
   (8.7/D9 prove it streams 14.5 GB/token; it must not be suggested).
3. Section 9.0 decoupling (separate change, ask before doing): pp device should join the scheduler
   backend list only, not `params.devices` (src/llama-model.cpp:1425-1490 side effect).
4. Untracked junk: `nul` and `bench_copy.obj` deleted already. `old_build/`, `old2_build/`,
   `*.log`, `*.bat`, `pp_*.md`, session html stay untracked (add to .gitignore only if asked).
5. Commit discipline: user writes/commits. Do not `git push`, do not open PRs. Per repo policy,
   open an issue describing the per-device host buft idea before any PR, it changes buffer
   ownership semantics for every `-ngl 0`/host-override config.## Risks / known unknowns

- AMD host-visible heap size on the 760M: verify `max_allocation_size` and heap flags cover a
  15.8 GB single buffer; if not, chunk weights into several buffers (loader already splits per
  tensor, only max-size per alloc matters).
- `GGML_SCHED_MAX_COPIES` / `n_copies`: with weights owned by Vulkan1 and decode on Vulkan1 there
  are no weight copies at bs=1, so `pipeline_parallel` interaction from the stash (forced off)
  should not regress decode; re-verify with the profile.
- If `select_weight_buft` (llama-model.cpp) prefers `Vulkan_Host` over the device buft for some
  override path, H2 must use the same selection as `-ngl 0` does today; check what D8/D9 actually
  exercised (`-ot ".*=CPU"` landed in Vulkan_Host, so the selection path exists).
- Upstream-relevant: this changes what `ggml_backend_dev_host_buffer_type` returns for integrated
  Vulkan GPUs globally (the 760M here, but every other Vulkan iGPU user too). Keep the discrete
  path byte-identical and document the UMA rationale for the issue.

---

## STATUS (updated in session)

Branch `pp-host-buft` (from `any-draft-device-fix`), pp-dev work stashed (stash@{0}), junk `nul`
and `bench_copy.obj` deleted. Phase 0 probe SKIPPED: existing measurements already cover both
questions (case C device bufts on the 760M are UMA `DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT`
mapped sysmem and matmul from them is proven at full speed; bench T1 measured the mapped-source
copy at ~11 GB/s), so Phase 1 was implemented directly.

### Phase 1 implemented (ggml-vulkan.cpp only, builds clean)

1. `vk_device_struct` gained `host_buffer_type` (per-device buft, context = regular
   `ggml_backend_vk_buffer_type_context`, name = "VulkanN_Host"), initialized next to
   `buffer_type` in `ggml_vk_get_device` (~line 7478).
2. Per-device host buft functions added after `ggml_backend_vk_buffer_type()`:
   alloc = `ggml_vk_create_buffer(device, size, {HOST_VISIBLE|HOST_CACHED|HOST_COHERENT,
   HOST_VISIBLE|HOST_COHERENT})` -> real vk_buffer, permanently mapped, `ptr` set by
   ggml_vk_create_buffer; alignment = minMemoryMapAlignment; max_size = suballocation_block_size;
   buffer iface = standard `ggml_backend_vk_buffer_interface` (so get/set/cpy/memset all work);
   is_host = CPU buft's (SEE CRASH below - this is wrong and will be flipped to NULL).
3. Interface `ggml_backend_vk_host_buffer_type_interface` defined next to the device-buft iface
   (top of file, before ggml_vk_get_device uses it). get_name REUSES
   `ggml_backend_vk_buffer_type_name` -> `ggml_backend_buffer_is_vk()` (compares fn pointers)
   accepts these buffers: descriptor binding, cpy_tensor, fused-op clone all work unchanged.
   supports_buft needs NO change (real context, device->idx matches).
4. Old global pinned buft kept as-is under `ggml_backend_vk_host_buffer_type()` with its functions
   renamed to `pinned_*` (it must keep the NULL context and distinct name fn
   `ggml_backend_vk_host_buffer_type_name` -> "Vulkan_Host"). Discrete GPUs still get this one.
5. `ggml_backend_vk_device_get_host_buffer_type(dev)` (was: ignored dev, returned global) now
   returns `&device->host_buffer_type` when `device->uma`, else the global pinned type.
6. `ggml_vk_buffer_copy` MULTI_DEVICE branch: if src is host-visible, skip the src GPU-copy
   stage and `ggml_vk_buffer_write(dst, ..., src->ptr + src_offset)` directly (dst-side staging
   memcpy + DMA). This also speeds up today's UMA device-buft -> V0 prefill path.

Smoke run (run_h1_smoke.bat, gemma, `-dev Vulkan1 -ngl 99 -ot ".*=CPU" --load-mode none`):
`load_tensors: Vulkan1_Host model buffer size = 3220.30 MiB` - weights DO land in the new buft,
selected automatically through make_cpu_buft_list (llama-model.cpp:1048 host entry) + the
`-ot .*CPU` path. Then:

### CRASH (must fix before continuing)

`llama_model_load: error loading model: read error: Invalid access to memory location.`
(= ERROR_NOACCESS 998, thrown by ReadFile in src/llama-mmap.cpp:126).

Root cause chain: --load-mode none -> llama-model-loader.cpp:1668 sees
`ggml_backend_buffer_is_host(cur->buffer)` == true (my buft set is_host = CPU's) and reads the
GGUF DIRECTLY INTO `cur->data`. But vk tensors never hold real pointers: vk's buffer_get_base
returns the fake base 0x1000 and real addresses are resolved per-op from bufctx->dev_buffer->ptr
(vk_tensor_offset = data - 0x1000). cur->data = 0x1000+off is garbage on Windows -> ReadFile faults.
The old Vulkan_Host never hit this because it was a genuine CPU buffer (cpu_buffer_from_ptr, real
base). Note `ggml_backend_buffer_get_base()` in ggml-backend.cpp:163 even GGML_ASSERTs is_host,
so a host buft with a fake base is an invalid combination by construction.

Fix (next step, small): set `is_host = NULL` in ggml_backend_vk_host_buffer_type_interface.
The buft then counts as a device buffer, exactly how vk compute treats it:
- loader:1668 falls to the else branch -> `ggml_backend_tensor_set(cur, ...)` ->
  ggml_backend_vk_buffer_set_tensor -> ggml_vk_buffer_write -> host-visible memcpy path (correct)
- loader:1610 stable_sort treats weights as staged (fine, bigger-first ordering, harmless)
- common/fit.cpp:78 memory accounting: buffers count against the device (cosmetically wrong on
  UMA but harmless; the memory-breakdown print will show them on Vulkan1)
- loader:1534 async-upload fast path: already disabled for non-default bufts (unaffected)
- clip.cpp:3617-style direct reads: mtmd only, not in this test path
After flipping, rerun run_h1_smoke.bat: SECOND FINDING below.

### SECOND FINDING: is_host=NULL trips the sched pairing assert

`GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b])) failed`
(ggml-backend.cpp:1896).

Cause: llama-context.cpp:436-441 pairs the (implicit last) CPU backend with
`ggml_backend_dev_host_buffer_type(model.devices[0])` as its scheduler buft. With the new
callback that returns the per-device VK host buft, whose is_host is now NULL, the CPU backend's
supports_buft (`buft_is_host`) is false -> assert. With is_host=true the same pairing worked
(first smoke attempt got all the way to weight loading), but is_host=true is what broke the
loader direct read. So:

- is_host=true  : sched pairing OK, loader crashes (fake base + direct ReadFile into data)
- is_host=NULL  : loader path correct, sched pairing assert (CPU backend + non-host buft)

Resolution (keeps vk semantics = device buffer, fixes the one bad pairing):
1. keep is_host = NULL in ggml_backend_vk_host_buffer_type_interface (DONE)
2. llama-context.cpp ~437: guard the CPU-backend pairing with
   `if (host_buft && ggml_backend_buft_is_host(host_buft))` so the CPU backend falls back to the
   plain CPU buffer type (same as it already does when a device has no host type, e.g. CUDA).
   Site 2 (~2167, output_dev host buft) can stay: an output buffer on the new buft behaves like
   any vk device buffer (fake base, logits read via tensor_get), same as CUDA today.
3. Everything downstream is then consistent: weights selected into Vulkan1_Host by
   select_weight_buft/make_cpu_buft_list; VK1 claims them in supports_buft; loader stages via
   ggml_backend_tensor_set -> buffer_set_tensor -> ggml_vk_buffer_write (host-visible memcpy,
   correct with fake base since offsets are buffer-relative); decode builds descriptors from
   bufctx->dev_buffer like any device buffer; get_base is never dereferenced.

Also noted: Vulkan1 now reports 48956 MiB total / 46508 MiB free (driver shared-memory sizing
varies by boot/load). Plenty for the 15.8 GB model in either heap. Borrowing machinery
(ggml_vk_tensor_subbuffer + device->pinned_memory registry) exists but is a different design
(needs real data pointers registered per allocation); not needed for this approach.

### THIRD FINDING: two separate crash paths, both from the new buft

A/B isolation: added env GGML_VK_NO_PERDEV_HOST_BUFT to the device callback (revert before
commit). With it set (= upstream behavior), case C on this branch runs fine
(gemma, gen 20.7 t/s). Without it:

- Case C regression (run_c_regression.bat): weights split V1 device 2362 MiB + Vulkan1_Host
  858 MiB (non-offloaded tensors land in the new buft via make_cpu_buft_list). Load dies with
  `read error: Invalid access to memory location` from ReadFile. Path: --load-mode none + all
  weights non-host -> loader async-upload fast path IS selected (bufs.at(0) is the V1 device buft
  == dev_buffer_type(dev)), and its pinned staging buffers are allocated from
  `ggml_backend_dev_host_buffer_type(dev)` = the new buft. host_ptrs = buffer_get_base(staging)
  = fake 0x1000 -> ReadFile into it -> AV. FIX: guard the async-upload path in
  llama-model-loader.cpp (~1553, after host_buft): require `ggml_backend_buft_is_host(host_buft)`
  (skip async uploads when the device reports a non-CPU-addressable host buft). Then the loader
  uses the plain staging path (tensor_set) which is correct for the new buft.
- H1 (all weights in the new buft): loads fine (async path correctly skipped there:
  Vulkan1_Host != dev_buffer_type(V1) -> upload_backend = nullptr), reserve succeeds with
  graph splits = 1 (whole graph on V1 - the goal state!), CPU gets an 8.3 MiB buffer for graph
  inputs only. Then AV at the first graph compute (with and without --no-warmup, right after
  `llama threadpool init`). NOT yet diagnosed. Next step: rerun with GGML_VK_DEBUG=1 and read
  the last VK_LOG_DEBUG lines before the fault; suspects: descriptor build vs my buffer's
  memory flags (HOST_CACHED path?), init_tensor ordering, or the CPU-backend graph-input copy
  (src is a real CPU buffer, cpy_tensor_async returns false at ggml_vk_host_get - that fallback
  is standard though).

State of the tree right now: per-device host buft active by default, is_host = NULL,
llama-context CPU-backend pairing guarded with ggml_backend_buft_is_host(), A/B env toggle in
the device callback (TEMPORARY), loader async-upload guard NOT yet added, H1 compute AV NOT yet
fixed. (Superseded by FOURTH FINDING below.)

### FOURTH FINDING: H1 crash root cause = fake vk base leaks into the output buffer

Bisection (TEMP prints): h1_nowarm.log ends exactly at "SCHED compute_async returned 0" for the
PP graph (2259 nodes, all completed, no VK final sync because AMD has support_async). The fault
is right after the PP graph compute returns: first logits access. llama-context.cpp:1520
`ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, ...)` memcpy's INTO
logits.data, and output_reorder (878+) std::swaps it - both treat it as a CPU pointer.

Root cause: llama_context::output_reserve (src/llama-context.cpp:2147) allocates buf_output from
`ggml_backend_dev_host_buffer_type(output_dev)` = the new Vulkan1_Host buft. logits.data /
sampling.* are raw pointers from `ggml_backend_buffer_get_base(buf_output)` (2161), but ALL vk
buffers return the fake tensor base vk_ptr_base = 0x1000 from get_base (ggml-vulkan.cpp:2571;
real addresses are per-op from bufctx->dev_buffer->ptr). First read of logits.data after the
first compute = AV. Upstream never saw this because its host buft (pinned CPU wrapper) returns a
real pointer. The is_host assert in ggml_backend_buffer_get_base is compiled out in release
builds, so get_base silently handed out 0x1000.

All three get_base consumers now guarded/fixed:
1. src/llama-context.cpp:2149 output_reserve: require ggml_backend_buft_is_host() on the output
   dev host buft, else fall back to plain CPU buft (same as discrete-GPU behavior; logits are
   then written by tensor_get). This was the H1 crash fix.
2. src/llama-model-loader.cpp:1553 async-upload staging: require ggml_backend_buft_is_host()
   else no async uploads (falls back to plain staging via tensor_set). Fixes the case C ReadFile
   AV from THIRD FINDING.
3. llama-context.cpp:436 CPU-backend pairing guard (already in place from earlier).

Checked safe: ggml-backend.cpp MoE expert-copy fast path requires is_host(input->buffer) - with
is_host = NULL it takes the normal async-copy branch. ggml_backend_buffer_clear on the new buft
goes to vk memset (fine, UMA branch memsets mapped memory directly).

Note for later: all host-visible vk buffers ARE persistently mapped in ggml_vk_create_buffer
(buf->ptr = mapMemory), so the real data pointer exists. A future optimization could give the
per-device host buft a real get_base (return bufctx->dev_buffer->ptr) so output_reserve and the
loader staging become zero-copy too. Deferred: init_tensor would also need to stop writing fake
tensor->data for these buffers.

TEMP debug prints all removed (ggml-backend.cpp split/compute_async prints, ggml-vulkan.cpp
graph_compute enter / node loop / final sync prints), GGML_VK_NO_PERDEV_HOST_BUFT A/B toggle
removed. is_host stays NULL. Tree rebuilt clean with build_pp.bat (only pre-existing warnings).

Memory flags: the per-device host buft alloc prefers coherent-only HOST_VISIBLE|HOST_COHERENT
with HOST_CACHED as fallback (same type case-C UMA device buffers get). Cached-first measured
together with a GPU contention artifact, coherent-first then re-measured clean: H1 = PP 74.1 /
TG 23.4, case C = PP 74.6 / TG 23.5 (gemma 4B, 30-token prompt, 16 gen). Case C exit=0 with the
loader guard - the ReadFile AV is gone. Both verified with llama-server.exe killed first
(benchmark hygiene: always taskkill //F //IM llama-server.exe before a timed run).

Phase 1 VERIFIED. Next: Phase 2 - stash pop (drop the stash's mmap_support=true hunk), rebuild,
H2 on Qwen3.8-27B: -dev Vulkan1 -ngl 99 -ot ".*=CPU" --load-mode none --pp-dev Vulkan0.

### FIFTH FINDING: my MULTI_DEVICE host-visible shortcut was the PP killer

Stash restore notes (for reference): `git stash pop` refused (same files dirty); `git apply --3way`
silently rolled back; what worked was `git apply --reject` (all hunks clean except arg.cpp +
common.cpp) plus `git checkout stash@{0} -- <files>` for the 8 files we had not touched ourselves
(arg.cpp, common.cpp, common.h, ggml-backend.h/.cpp, llama.h, llama-context.h, llama-cparams.h).
llama-context.cpp pp_backend hunks had already landed from the aborted --3way. The stash's
mmap_support=true hunk was applied by --reject and then reverted by hand (handoff 2.1).

H2 first run: PP 1.48 t/s. GGML_SCHED_PROFILE showed the split: PP graph = 13.2 GB weight copies
in 91-181 s (~100 MB/s) vs compute 0.6-1.6 s. Cause: the Phase-1 MULTI_DEVICE shortcut in
ggml_vk_buffer_copy (memcpy from mapped host src) CPU-READS uncached UMA sysmem = ~100 MB/s.
The upstream staging round trip (V1 GPU copy to staging, CPU memcpy staging-to-staging, V0 GPU
write) never CPU-reads weight data: 4.5-5.5 GB/s. Shortcut removed (with a comment); copies now
2.4-2.9 s per 353-token graph.

### H2 results (Qwen3.8-27B-MXFP4, c=85248 b=ub=1024, q4_1 KV, fa, server killed first)

| config | PP t/s | TG t/s | notes |
|---|---|---|---|
| A baseline (8.1b, weights V1 device buft) | 58.8 | 3.75 | reference |
| D pp-dev (8.1b, weights V1 device buft) | 73.5 | 2.98 | same 1.1k splits, same copy path |
| streaming (8.9, MIN_BATCH=1) | 35.6 | - | 14.5 GB/token |
| H2 no-pp-dev: weights V1_Host, graph splits = 1 | 40.5 | **4.16** | best TG ever, zero copies |
| H2: weights V1_Host + --pp-dev Vulkan0 | **47.5** | 2.13 | 1150 splits (PP) / 134 (decode) |

Readings:
1. All 15.8 GB land in Vulkan1_Host, load clean, decode-only graph is one split on V1 and beats
   every previous config (4.16). The host-buft concept is proven at 27B scale.
2. --pp-dev PP 47.5: compute runs on V0 (353-token profiled graph: compute 0.5-1.5 s only), but
   the 13.2 GB per-batch weight copy (2.4-2.9 s, synchronous per copy: two GPU submits + fence
   waits + CPU memcpy) eats the margin. Old D got 73.5 because its copies came from the V1 device
   buffer... same sysmem, same path - the remaining D vs H2 gap is unexplained; candidates:
   per-copy fence sync (1511 copies x submit+wait), V0 holding a 104 MiB KV slice (D had 64),
   and HOST_COHERENT-only source (flipped from cached for TG; may slow the GPU-side copy reads).
3. TG 2.13 vs 4.16 without pp-dev: decode graph gets 134 splits and 197 activation copies because
   V0 holds a KV slice + split overhead. Same qualitative hit as 8.1b D (2.98 vs 3.75), larger.
4. Next lever for PP: zero-copy prefill by importing the V1_Host allocation into V0
   (VK_EXT_external_memory_host import path already exists in ggml_vk_create_buffer via
   import_ptr; ggml_vk_tensor_subbuffer + device->pinned_memory registry). Register each
   Vulkan1_Host buffer once as a V0-imported buffer, then PP ops on V0 bind the same physical
   memory. That removes the whole 13.2 GB/batch copy. Bigger change, needs design care.
5. TG lever: keep V0 out of decode. Check why 104 MiB of KV lands on V0 (no V0 model buffer is
   logged, so dev_layer says V1; suspect reserve-time forced assignment or gpu buft list order),
   or evaluate whether server use cases accept pp-dev TG hit for long-prompt PP win.

### Then Phase 2 (unchanged from plan above)

1. `git stash pop` (stash's ggml-vulkan hunk is just mmap_support=true at 19336 - DROP it,
   handoff 2.1 says it must be reverted; resolve conflict by keeping our file's false).
   Restores: forced-offload branch (offload_backend_id + min_batch 32), --pp-dev plumbing,
   pipeline_parallel guard, GGML_SCHED_PROFILE, copy_lookahead (keep dormant, env-gated).
2. H2: `-dev Vulkan1 -ngl 99 -ot ".*=CPU" --load-mode none --pp-dev Vulkan0` on Qwen3.8-27B
   (kill llama-server first, handoff 4.2): expect PP ~110-125 (fast single-staging copy via the
   new MULTI_DEVICE shortcut), TG ~3.5+ (zero-copy decode, weights now V1-owned).
3. H3: same WITHOUT `-ot` -> with mmap default the loader redirects to plain CPU buft, so H3 must
   use `--load-mode none` too; verify whether plain `-ngl 99` alone (no -ot) also lands in
   Vulkan1_Host now (it should NOT: -ngl 99 picks the device buffer_type for weights, which is
   still the DEVICE_LOCAL|HOST_VISIBLE UMA type - i.e. baseline case C unchanged, good).
4. Regression: gemma case A (`-dev Vulkan1 -ngl 99`, no -ot) must be byte-identical behavior to
   before (device bufts untouched); gemma `-ngl 0` now goes through the new host buft - compare
   PP/TG vs handoff section 11 rows C/D.

### SIXTH FINDING: V0 gets 8 layers (and their KV) because --pp-dev APPENDS it to the device list

h2_qwen.log: base KV cache (85248 cells, 16 full-attn layers, 1665 MiB total) is split
Vulkan0 104.06 MiB + Vulkan1 1560.94 MiB. 104.06 = exactly 1/16. Cause: common.cpp pp_dev block
APPENDS Vulkan0 to params.devices -> 2-GPU model. Default split is by free memory (llama-model.cpp
1459: splits[i] = free; V1 reports ~46 GB free UMA, V0 ~5 GB) -> normalized split point ~0.90 ->
get_layer_buft_list upper_bound assigns the LAST layers (61-65) to V0. The -ot ".*=CPU" override
moves their WEIGHTS into Vulkan1_Host, but layer ASSIGNMENT still controls: KV cache buft
(llama-kv-cache.cpp 214: dev_layer(il) -> V0 for those layers) and the graph callback
(llama-context.cpp:2557 n_tokens<32 branch) pins norm/l_last onto dev_layer(il) = V0. That is the
134 decode splits and the 2.13 TG (vs 4.16 without pp-dev).

FIX (designed, not yet applied): in the common.cpp pp_dev block, when pp_dev was appended (it was
not in the user's device list) and the user gave no explicit -ts, set
tensor_split[idx_of_pp_dev] = 0.001f (and leave others 0 -> not all_zero -> splits taken from
tensor_split directly; 0.001/total ~ 0.00007 -> upper_bound assigns 0 layers to V0; the epsilon
instead of 0 protects the all_zero==true fallback and division). Do NOT touch anything when the
user passes -ts explicitly (respect user). After fix, expect: no Vulkan0 KV buffer line,
splits bs=1 = 1, TG = 4.16.

### OPEN ISSUE 7: PP rate decays during long prompts; iGPU 90% / dGPU 5% during them

Symptom (server, long prompts): PP starts 100+ t/s then drops sharply. Task manager: iGPU ~90%,
dGPU ~5% during long prompts.

Data we have (h2_qwen.log, 1386 tok): chunk1 32.6 t/s -> chunk2 59.0 t/s (cumulative avg
converging up to the 47.5 final), no decay visible at this prompt length. The decay is only seen
on long server prompts, so we have NO per-ubatch data for one yet. Needed first: a controlled
long-prompt run (server or cli, ub=1024, lv 5) capturing (a) the per-ubatch
"prompt processing, n_tokens = N, progress, t/s" lines, (b) nvidia-smi dres and task-manager iGPU
samples every few seconds, (c) GGML_SCHED_PROFILE=1 for copy/compute split per ubatch.

Hypotheses, ordered:
1. iGPU 90% = the iGPU is doing the weight copies / staging memcpys (its 3D/Video engines or
   compute queue runs the staging copies), NOT the matmuls (compute is on V0). As the prompt
   grows, KV-cache reads/writes and the growing tail of V0-owned layers move more work to V1.
2. Checkpoint growth: "created context checkpoint N of 32 ... size = 149.626 MiB" - up to 32 x
   150 MiB ~ 4.8 GB of seq-state buffers are allocated on the KV devices as the prompt grows
   (llama-context.cpp 2053 out_ids copying, create_check). On a 6 GB V0 that could evict/slow
   allocations mid-prompt; on V1 (UMA) it competes with the 15.8 GB host weights for bandwidth.
3. UMA bandwidth contention: as KV grows, every ubatch both copies 13.2 GB of weights (V1->staging
   sysmem traffic) AND streams more KV cells; both hit the same 62 GB shared DRAM. iGPU saturates,
   PP decays. dGPU idles waiting for copies (5%).
4. Thermal/power shift on the APU (760M shares power budget with CPU); long sustained copy loads
   downclock the iGPU. Check with a sensors log during a long run.
5. Scheduler split explosion with growing n_kv: split count for PP graph can grow with context
   (kv cache views/copies per split); check "graph splits" in server logs at different prompt
   lengths.

Investigation plan (next session):
1. Reproduce with logs: llama-server (same params as H2) + a 20-30k token prompt; log timestamps
   of each ubatch progress line; sample nvidia-smi + iGPU usage concurrently.
2. From h2_prof.log methodology, run GGML_SCHED_PROFILE=1 on a 4k prompt cli run to see if copy
   time per ubatch grows with position (KV traffic) or stays flat (then it is thermal).
3. Check checkpoint allocation: grep "created context checkpoint" counts + sizes; try -ck 0 or
   the server flag to disable checkpoints (cache_reuse) to test hypothesis 2 directly.
4. If copies grow with position: the KV copy cost is per-split; after SIXTH FINDING fix re-measure
   (V0 out of decode may also change the PP split topology).

### SEVENTH FINDING (ISSUE 7 SOLVED): decay = FLASH_ATTN_EXT stuck on the KV device

SIXTH FINDING fix applied first (epsilon-free tensor_split [1,0] in common.cpp pp_dev block,
guarded by pp_appended && !ts_set && n_devs>=2): Vulkan0 KV buffer gone, decode splits 134 -> 97,
TG 2.13 -> 3.07, PP unchanged. All layers now on V1.

Controlled 12k-token run (h2_long.log, pre-fa-fix): per-chunk rates 93.7 -> 89.4 -> 86.3 -> 82.7
-> 79.3 -> 68.3 -> 54.8 t/s. dGPU 0-9% util during PP. GGML_SCHED_PROFILE attribution per ubatch:
copy FLAT at 4.22 s (13.2 GB, position-independent), compute grows LINEARLY 0.5 -> 9.5 s. Sched
dump (GGML_SCHED_DEBUG=2): all 272 FLASH_ATTN nodes on Vulkan1, cause 2.sup (KV cache lives
there), while matmuls are forced to V0. FA has no weight sources, and ggml-backend.cpp:980 skips
it from the weights rule ("sinks tensor too small"), so attention follows the KV cache and stays
on the iGPU at ANY batch size. Attention cost grows with n_kv -> linear compute growth -> decay,
iGPU saturation (90%), dGPU idle (5%).

FIX: in ggml_backend_sched_backend_id_from_cur (after the FA skip), force FA to the offload
backend for prefill-sized batches (op_batch_size >= offload_backend_min_batch; FA rows =
n_tokens so decode bs=1 is unaffected). Cost: FA reads K/V views from the V1 KV cache, so each
layer's K/V (~1.7 GB/ubatch at 12k ctx, q4_1) is copied V1 -> V0; trivial vs the 13.2 GB weight
copies and it removes every per-layer V1 compute island from the PP graph.

RESULT (h2_long.log after fix, same 12k prompt, ub=1024):
- PP overall 122.0 t/s (was 83.8), per-chunk FLAT ~152 t/s instant, no decay (last cumulative
  dip = final chunk includes decode + checkpoints)
- TG 3.41 t/s (was 3.07)
- splits bs=1024: 1186 (FA/KV copies add boundaries, but every split now computes on the fast
  device), bs=1 still 97
- exit 0, sane output with q4_1 KV + FA on V0
- safety: the new branch requires offload_backend_id >= 0, so runs without --pp-dev are
  byte-identical (H1, case A/C unaffected); decode unaffected by the min_batch gate

Remaining: TG 3.41 vs 4.16 no-pp-dev (97 decode splits from the norm/l_last callback pins -
separate, small); zero-copy prefill via external memory import would remove the 13.2 GB copies
and could push PP well past 150.

### BACKLOG (user requests)

1. CUDA0 as the prefill device - CLOSED, see "CUDA0 as pp device: closed" below. Hypothesis was
   that the weight copies would ride the generic scheduler path (sysmem -> CUDA staging -> H2D DMA)
   instead of the Vulkan0 staging round trip, and come out faster. Measured: 74.0 vs 132.5-134.9
   t/s. It does not. Vulkan0 stays.
2. MTP-on-dGPU as the standard layout (matches the user's normal 27B dense setup: iGPU = main
   device, dGPU = MTP draft / DSpark when adventurous). Current optimal.bat already does this via
   --spec-draft-device Vulkan0. Validate the MTP+pp-dev combination end-to-end (draft graph and
   verify graph interplay with the forced-offload, esp. that verify stays on V1 with default
   MIN_BATCH=32), then check DSpark compatibility. Done for MTP in the real-work log: draft
   graphs were the small 15-split ones, healthy; only verify was distorted by MIN_BATCH=1.

### CUDA0 as pp device (backlog item 1, first results)

Tree rebuilt Vulkan+CUDA (user's flags, sm_89, configure_pp.bat). MTP A/B via llama-cli
(ab_on/ab_off.log, 353 tok, n=128, draft on V0): PP 43.5/43.5, TG 6.40 (MIN_BATCH=1) vs 6.63
(default) -> default wins, MTP healthy, +59% over no-spec 4.16. 12k ctx config B: PP 134.9,
TG 4.76. User's real-work 2.51 TG was MIN_BATCH=1 (verify on V0, KV streamed over PCIe) plus
GGML_SCHED_PROFILE=1; optimal.bat fixed.

CUDA0 pp-dev results, Qwen 12k prompt:
- first run (coherent-only V1_Host): PP 51.9 t/s, TG 3.35. Copy path = generic sched copy ->
  get_tensor on the V1_Host buffer -> CPU read of UNCACHED UMA sysmem (the FIFTH-finding trap in
  a new place; with a Vulkan pp dev the copies never CPU-read).
- fix: ggml_vk_buffer_read_2d routes reads > 16 MiB from uncached host-visible UMA memory
  through the GPU staging (device copy to cached staging, then CPU memcpy from cached).
- host buft flipped back to cached-first (HOST_CACHED|HOST_COHERENT preferred, coherent-only
  fallback): CUDA0 PP 74.0, Vulkan arm unchanged (PP 132.5 / TG 4.63 vs 134.9/4.76 = noise).
  Cached-first kept: strictly better for cross-device copies, no Vulkan cost. NOTE: with cached
  memory the >16MiB staging detour in buffer_read_2d is bypassed (memory IS cached -> direct
  memcpy at DRAM speed), so the two changes compose correctly.
- copy_lookahead test: PP 81.5 but crash at first decode (llama-sampler.cpp:1211 assert, garbage
  probs -> lookahead produces wrong logits). Confirms handoff 8.10: stays env-gated OFF.

Superseded by the close-out below: the 74 vs 135 gap was diagnosed as a copy-path problem, and the
direction was closed because the only real fix is a new cross-backend contract that Vulkan0 does
not need.

### User's real-work observation (llama-atest.log, resolved)

Decode profiled cycle: 3x small (15 splits, ~33 ms, MTP draft on V0) + 1x big (195 splits,
~250 ms copies + ~880 ms compute). Cause: GGML_OP_OFFLOAD_MIN_BATCH=1 in the user's bat -
stale from the old device-order mechanism, it forced verify (4-token batch) onto V0 too, so the
verify graph ping-ponged and streamed KV over PCIe (user saw 200+ MB bursts in HWiNFO; iGPU 80% /
dGPU 20% during decode). GGML_SCHED_PROFILE=1 was also on in a real-work run (serializes, skews).
Fix: drop both env vars - captured in optimal.bat (project root). Everything else in the user's
bat was already correct.

### CUDA0 as pp device: closed

The gap (CUDA0 PP 74, Vulkan0 PP 135) is not a kernel issue, it is the copy path. Per weight tensor
copied Vulkan1_Host -> CUDA0 in `ggml_backend_sched_compute_splits`:

1. the dst backend's `cpy_tensor_async` is tried first (ggml-backend.cpp:1890).
   `ggml_backend_cuda_cpy_tensor_async` (ggml-cuda.cu:2477) returns false unless BOTH sides are CUDA
   (line 2481), so a Vulkan source can never use it.
2. fallback (ggml-backend.cpp:1891-1899): synchronize the Vulkan source, then synchronize or wait on
   the CUDA destination, then a blocking `ggml_backend_tensor_copy`.
3. `ggml_backend_tensor_copy` (ggml-backend.cpp:488) sees neither side as host (`is_host` is NULL on
   the per-device host buft by design), so it calls `ggml_backend_buffer_copy_tensor` ->
   `ggml_backend_cuda_buffer_cpy_tensor` (ggml-cuda.cu:820), which also requires a CUDA source and
   returns false.
4. last resort (ggml-backend.cpp:503-507): `malloc(8.7 MiB)` + `ggml_backend_tensor_get` (Vulkan
   reads the mapped host buffer on the CPU) + `ggml_backend_tensor_set` (cudaMemcpyAsync H2D from
   pageable malloc memory, then cudaStreamSynchronize).

So each of ~1500 weights per ubatch pays two backend syncs, a malloc, a CPU read and a pageable H2D
copy, all serialized.

Why Vulkan0 does not have this: there the destination is Vulkan, so
`ggml_backend_vk_buffer_cpy_tensor` (ggml-vulkan.cpp:17103) takes the `is_vk(src)` branch. The
per-device host buft reuses `ggml_backend_vk_buffer_type_name` as `get_name`, so a Vulkan1_Host
source IS a vk buffer and the copy becomes `ggml_vk_buffer_copy` MULTI_DEVICE staging (GPU copy to
staging, CPU staging-to-staging memcpy, V0 DMA write). No malloc, no pageable H2D, no cross-backend
serialize.

Root blocker: vk tensors have a fake base (`vk_ptr_base = 0x1000`); the real address lives in
`ggml_backend_vk_buffer_context::dev_buffer->ptr`. No interface lets a foreign backend ask for it, so
CUDA cannot DMA from the weights.

Measured comparison, Qwen3.8-27B MXFP4, 12k prompt, q4_1 KV, fa:

| metric | CUDA0 | Vulkan0 |
|---|---|---|
| PP t/s | 74.0 best (51.9 with coherent-only host memory) | 132.5-134.9 |
| weight copy per tensor | 2 backend syncs + malloc + CPU read + pageable H2D, serialized | GPU staging copy, no malloc, no pageable H2D |

Reasons the direction is closed, in order of weight:

1. It is already 1.8x slower on the metric it was meant to win. Nothing in the diagnosis suggests a
   cheap change that recovers that.
2. The only real fix is invasive. (A) below needs a new optional accessor on the shared buffer
   interface, a `cudaHostRegister` on a Vulkan-owned allocation, and a CUDA copy entry point that
   accepts a foreign source. That is a new cross-backend contract spread over
   ggml-backend-impl.h + ggml-cuda.cu + ggml-vulkan.cpp. Per AGENTS.md it needs a prior issue, and the
   case for it is weak while the Vulkan path already works.
3. The upside is bounded. Pinning removes the staging and the CPU touch, and CUDA's separate copy
   engine makes real overlap possible where Vulkan is single-queue on this card (8.10) - but 15.8 GB
   per ubatch still has to cross PCIe. Best case it trades an invasive rework for roughly what Vulkan0
   already delivers.
4. The original goal is already met: zero-copy decode on the iGPU plus 131.5 t/s prefill.

Alternatives considered, kept as the record rather than as pending work:

- (A) expose the real host pointer: new optional buffer accessor implemented by the vk host buft, plus
  `cudaHostRegister` on the mapped allocation, plus a CUDA copy entry point accepting a non-CUDA
  source. This was the only route that could beat 135 t/s. Not implemented.
- (B) pin the weights with cudaMallocHost and import into Vulkan via VK_EXT_external_memory_host.
  BLOCKED by hardware: the AMD driver rejects host-memory imports (handoff 8.2,
  ErrorInvalidExternalHandle).

Revisit only if CUDA is wanted for a capability reason (a quant type or op that Vulkan lacks) rather
than for throughput. Note the ub>=3840 crash is device-independent, so CUDA0 would also be capped at
ub=3072 for this model even if (A) were done.

### Audit: is_vk callers and get_base consumers (clean)

`ggml_backend_buffer_is_vk` now also matches the per-device host buffers (it compares the `get_name`
function pointer, which they reuse on purpose). All callers stay correct: the buffer cpy_tensor
(ggml-vulkan.cpp:17103) and cpy_tensor_async (17470) only use `bufctx->dev_buffer` and
buffer-relative offsets, and every other caller is inside GGML_VULKAN_CHECK_RESULTS, which is off.
The new behavior is a win: it is what enables the Vulkan1_Host -> V0 staging copy in the first place,
and it also gives same-device HOST_VISIBLE -> device copies the async vk copy path.

get_base consumers, all safe:
- `llama-context.cpp:2172` output_reserve: guarded by `ggml_backend_buft_is_host`, so logits fall back
  to the plain CPU buffer (the FOURTH FINDING fix).
- `llama-model-loader.cpp:1560` async-upload staging: guarded, so async uploads are skipped.
- `llama-model.cpp:1781` mlock: requires `ggml_backend_buffer_is_host`, which is false for the new
  buft, so mlock is skipped. Safe by construction, no change needed.
- `llama-model.cpp:1909` / `llama-kv-cache.cpp:691` (the `no_alloc` assert base == nullptr): these
  compare against `vk_ptr_base` for any vk buffer, so they are pre-existing behavior for Vulkan
  device bufts too, not introduced here.

### Branch hygiene (done)

The tree had a large number of local artifacts accidentally staged. Unstaged (left on disk as
untracked): the session html, `.pi/continue/*`, `pp_chunked_*.md`, `pp_debug_next_steps.md`, the
`prompt*.txt` / `test_prompt.txt` inputs, `test_decode_results.txt` and the `test_pp_*` results and
scripts. Removed the stray `nul` file. `tasks.md` and `handoff.md` stay tracked. The only tracked
change now is a comment fix in ggml-vulkan.cpp (see below).

The cached-first A/B note-to-self in `ggml_backend_vk_host_buffer_type_alloc_buffer` was replaced by a
statement of the decision and the reason. The coherent-only arm stays as the driver fallback, so the
`uncached_uma` staging detour in `ggml_vk_buffer_read_2d` is NOT dead code - it exists for drivers
without a cached host type.

### ub=4096 crash (Qwen3.8 / QWEN35): graph reservation, QWEN35-specific

The sweep's b4096/ub4096 rows are missing because the process segfaults (exit 139) during
`sched_reserve`, before any buffer-size line is printed. Reproduced with `-fit off`, and with every
weight placement (device buft, Vulkan1_Host, with and without `--pp-dev`), so it is not caused by the
pp-dev feature.

Measured threshold, Qwen3.8-27B Q4_K_XL, `-c 5000`, q4_1 KV, `-fa on`, MTP on V0 (sweep config):

| ub | result |
|---|---|
| 512 | OK, Prompt 75.5 (sweep) |
| 1024 | OK, Prompt 106.1 (sweep) |
| 2048 | OK, Prompt 120.2 |
| 3072 | OK, Prompt 131.5 |
| 3840 | SEGFAULT (139) |
| 4096 | SEGFAULT (139) |

Control: gemma-4-E4B at ub=4096 runs fine (Prompt 323.3, exit 0), so this is QWEN35-specific and not a
general ubatch limit.

Crash location: `sched_reserve` -> `resolve_fused_ops` (llama-context.cpp:665) -> the DeepSeek V4 HC
probe -> `graph_reserve(1, 1, 1, mctx, true)`. The last logged line is the probe's graph reserve, then
the process dies with no error message, which fits an out-of-bounds access with the NDEBUG asserts
compiled out.

Suspect, NOT proven: `llama_context::graph_max_nodes` (llama-context.cpp:2351) gives QWEN35
`n_tokens * 40` nodes, and the KIMI_K3 branch directly above carries the comment "the n_tokens*40
budget below is exhausted at ubatch 3840" and uses `*160`. Our threshold sits exactly there. But a
linear requirement `a*n + b` against a `40*n` budget can only fail always or never, never above a
threshold, so node exhaustion cannot be the mechanism on its own.

Next step: a stack trace (WinDbg/cdb, or a debug build) for the faulting frame. Note `auto_fhc` is on
by default (llama-context.cpp:242) and has no CLI flag, so the DeepSeek V4 HC probe runs for every
model. Practical ceiling today: ub=3072 at 131.5 t/s, +9% over ub=2048 and the best prefill number
measured on this setup.

