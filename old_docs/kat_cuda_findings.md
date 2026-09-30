# KAT-Coder CUDA0 slowdown: root cause and fix

Date: 2026-09-28. Model: `Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf` (21 GB, 41 layers, MoE).

## Symptom

Same model, same CUDA0: "regular" build 20+ t/s gen, our branch build ~6-7 t/s.

## Root cause

Not the build. Both builds are byte-for-byte comparable in each mode. It is the
invocation:

- with `--lazy-mode on` (+ explicit ngl), fit-params ABORTS ("n_gpu_layers already
  set by user") and weights stream from mmap over PCIe on every token: ~7 t/s.
- without lazy, fit-params runs and finds the good layout: dense weights in VRAM,
  38 of 41 layers' expert weights overflowed to system RAM
  ("41 layers (38 overflowing), 3945 MiB used"): ~22-27 t/s.

Measured matrix (prompt t/s / gen t/s, -c 2048 -b 2048 -ub 2048 -fa on,
GGML_CUDA_NO_PINNED=1, 128 tok, machine otherwise idle):

| config | llama-pp build (Sep 23) | branch build (Sep 27/28) |
|--------|--------------------------|--------------------------|
| lazy on + -ngl 99 | 24.6 / 7.0 | 23.9 / 7.0 (also 6.6 with -fa on) |
| fit active (no lazy, no ngl) | 20.1 / 21.9 | 36.6 / 26.9 |
| router + preset (fit path) | ~20+ (daily use) | 31.0 / 25.5 |

The contaminated first measurement of the old-build fit path (2.2 t/s) was caused
by a leftover router instance holding 2.9 GB VRAM during fit; rerun clean = 21.9.

## Why it looked build-related

The daily router preset has no `lazy-mode` key, so the router always took the fit
path. Any manual/CLI test that copies the preset and adds `--lazy-mode on` (e.g.
to avoid the Vulkan OOM on Flash-Next) silently kills fit and drops to 7 t/s.

## Fix / guidance

- KAT on CUDA0: do NOT set lazy-mode. Let fit-params place expert weights in RAM.
- Keep `GGML_CUDA_NO_PINNED=1` (no effect measured here, but harmless and it is
  what the daily bat sets).
- If a preset needs lazy for loading (Flash-Next Vulkan), expect the fit path to
  be unavailable: that is the slow-7-t/s tradeoff, not a bug.

## Open question for later review

lazy AUTO + fit: could fit's overflow layout and lazy mmap be combined so the
7 t/s path also benefits? Fit aborts because ngl is "already set by user" whenever
lazy is on; possibly fit should treat lazy-on as "layout free, capacity unlimited"
instead of aborting. Left as a design question - not changed in this session.

## Sweep bat mystery (resolved)

The CUDA sweep failed 2026-09-28 morning with "invalid device: CUDA0" from every
run. Cause found at 13:xx: `test_v1sc_split.bat` had `set BIN=D:\progs\llama-cpp-adv`
- the Vulkan-only install, which cannot see CUDA0. This edit was not made by the
agent in this session (user review requested). Restored to
`D:\shared\project\llama.cpp\build\bin` (fresh GGML_CUDA=ON build) and relaunched.
