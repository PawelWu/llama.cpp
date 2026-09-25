# Tasks - per-device host-visible buffer type (`pp-host-buft`)

Branch: `pp-host-buft`, created from `any-draft-device-fix` (b1cb4909c).
Parent work stashed on `pp-custom-device` (stash@{0}, "pp-custom-device WIP: forced offload branch ...").
Note: commit 25515e4c7 (`--pp-dev` arg) is NOT in this branch. It comes back with the stash.

Context: handoff.md sections 3 (corrected), 8.7, 8.8, 8.9. Read those first.

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

