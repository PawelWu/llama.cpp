# old_docs

Working notes and findings from finished or superseded work. Kept for the numbers,
log names and dead ends so they do not have to be re-derived. Current state of the
branch lives in `../tasks.md` (live log) and `../handoff.md` (long-form reference,
section 0 first). Raw sweep logs and summaries live in `../prompts_perf/` and `../old_logs/`.

## Archived 2026-09-30

| file | what it is | status |
|---|---|---|
| `pp_chunked_debug_state.md` | chunked prefill NaN debug state (Sep 13) | OBSOLETE - chunked/streaming prefill path was abandoned for the per-device host buft; see handoff section 0 and 10 |
| `pp_chunked_solutions.md` | chunked decode NaN hypotheses to test (Sep 14) | OBSOLETE - same as above, never resolved and no longer relevant |
| `pp_debug_next_steps.md` | chunked decode NaN debug options (Sep 12) | OBSOLETE - same as above |
| `pp_dev_ot_cpu_fix.md` | why pp-dev + `-ot ".*=CPU"` was 3x slower at TG: `--load-mode mmap` silently discards the `-ot` host placement (Sep 29) | DONE - guidance folded into handoff/tasks; use `--load-mode none` for any `-ot` test |
| `pp_tg_gpu_attribution.md` | where the old build's 15+ GB of RTX shared VRAM came from, and what V0 does during TG (Sep 29) | DONE - old-build artifact; per-device host buft removed it (302 MiB). gpumon usage notes still valid |
| `pp_tg_trace_plan.md` | plan for per-split PP/TG timeline tracing (Sep 29) | NOT DONE - kept as the design if trace tooling is ever picked up again |
| `kat_cuda_findings.md` | KAT-Coder on CUDA0: `--lazy-mode on` aborts fit-params, 7 t/s vs 22-27 (Sep 28) | DONE - guidance: on CUDA0 do not set lazy-mode, let fit place expert weights in RAM |
| `test_27b_ud_split.md` | sweep doc: 27B UD-Q4_K_XL split + MTP placement (Sep 28) | RUN - results in `../prompts_perf/sweep_27b_ud/`; superseded by the pp-host-buft approach |
| `test_v1sc.md` | sweep doc: CUDA0,Vulkan1 split + the 40 GB CUDA0 WDDM oversubscription incident (Sep 28) | RUN - results in `../prompts_perf/sweep_v1sc/`; rule kept: lazy + user `-ts` on a 6 GB card is unusable |
| `flashnext_tg_analysis.md` | Flash-Next 47 GB MoE: why TG is low, traffic model, t1/t2 split plan (Sep 28) | PARTIAL - t1 measured (iGPU compute-bound, 5.6 t/s); t2 (dense on 4050) never run. Revisit from section 5 if Flash-Next TG work resumes |
| `freebuff-chat-2026-09-27T11-48-24.791Z.md` | exported agent session log (Sep 27) | REFERENCE ONLY - raw transcript, nothing unique outside tasks.md/handoff.md |
