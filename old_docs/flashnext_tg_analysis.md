# Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS: why TG is low, and which split can raise it

Target: more TG (generation), PP is accepted as-is.
Model: `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` + `-00002-of-00002.gguf`
Build: `build/bin` (CUDA + Vulkan, sm_89), `GGML_CUDA_NO_PINNED=1`
All tensor numbers below are read from the load logs in `prompts_perf/sweep_v1sc/`, not estimated.

## 0. What the model is actually made of

Arch `qwen4exp`: 48 layers, n_embd 2560, n_head 24, head dim 256, 512 experts with 10 active.
Hybrid: every layer has `attn_qkv`, 36 of 48 layers additionally have an SSM block (`ssm_*`;
layers 3, 7, 11, ... 47 are attention only). There is a PLE (per-layer n-gram embedding) table,
`per_layer_token_embd`, created with `TENSOR_READ_LAZY` (`src/models/qwen4exp.cpp:187`).

Weight inventory (part1 = 44850 MiB, part2 = 27465 MiB, total 70.6 GiB):

| class | size | note |
|---|---|---|
| MoE experts (`ffn_{gate,up,down}_exps`) | **40888 MiB** (851 MiB/layer) | gate 271 + up 271 + down 309 MiB per layer |
| dense, all layers (`attn_qkv`, `attn_gate`, `attn_out`, `ssm_*`, `hc_*` x4, `ffn_gate_inp`, `ple_*`) | **2914 MiB** (60 MiB/layer) | e.g. blk.0: attn_qkv 13, attn_gate 6, ssm_out 7, hc_* 4x6, gate_inp 2 |
| `token_embd` | 260 MiB | iq3_s |
| `per_layer_token_embd` | 27465 MiB | part2, lazy host buffer, gathered per token via `ggml_get_rows` |
| shared-expert tensors (`ffn_*_shexp`) | ~0 MiB | present but empty in this quant |

So: **experts are 93% of the layer weights, the always-read dense set is 2.85 GiB**, and the 27 GiB
PLE table is a capacity problem only (a per-token gather of `ple_n_heads` rows per PLE layer, tens of
KB of traffic, `src/models/qwen4exp.cpp:1024`).

## 1. Per-token decode traffic (the TG cost model)

| traffic | bytes/token | readable at |
|---|---|---|
| dense 2.85 GiB, read in full every token | 2.85 GiB | 200 GB/s in dGPU VRAM, ~55 GB/s in iGPU UMA, disk speed if paged out |
| experts, 10/512 active = 1.95% of 39.9 GiB | 0.78 GiB | same, but reads are scattered across a 39.9 GiB set |
| KV q4_1 (136 MiB at ctx 20k, ~580 MiB at 85k) + RS state 112 MiB | ~250 MiB | wherever the layer split put it |
| PLE gather | ~KB | host mmap |

Note the ratio: **the dense set is 3.6x the per-token expert traffic.** That is the opposite of what
the current `-ts`/`ngl` layer split assumes. A layer split spends VRAM on whole layers, i.e. mostly on
expert tensors that are read at 1.95%, while the tensors that are read 100% of the time are 6% of the
model. That is the main structural mistake to fix for TG.

## 2. Why TG is 1.5-4.2 t/s today

At decode (`bs=1`) the scheduler weight-offload rule does not fire: `ggml_backend_offload_op` is gated
at batch >= 32 in both backends (`ggml/src/ggml-backend.cpp`, `ggml-vulkan.cpp:20135`). A weight that
sits in a host buffer therefore has its op **executed on the CPU** at decode; nothing is copied to a
GPU. Decode throughput is decided purely by where each weight physically lives:

| measured config | where the weights are | prompt | gen |
|---|---|---|---|
| c0h `-dev CUDA0 -ot .*=CPU` | all 43 GiB mmap host | 31.9 | 1.7 |
| c0l `-dev CUDA0 --lazy-mode on` | dense on CUDA0 2.4 GiB, experts host | 25.8 | 1.8 |
| s0h `-dev Vulkan0 -ot .*=CPU` | all mmap host | 176.2 | 2.9 |
| s0l `-dev Vulkan0 --lazy-mode on` | all mmap host | 194.2 | 1.5 |
| s15cm `-dev Vulkan0,Vulkan1 -ts 85,15 --lazy-mode on --cpu-moe` | dense 2.8 GiB on devices, experts host | 118.2 | 4.2 |
| preset `C0+V1` (`ts 23,8`, lazy) | asks 43 GiB on CUDA0 | - | OOM (see `test_v1sc.md`) |

Readings:

1. Every config that keeps the expert set in the mmap host buffer pays for it: 39.9 GiB of expert pages
   cannot stay in the page cache next to a 27 GiB PLE file and the rest of the system, so the 0.78 GiB
   slice a token needs is a mostly cold read. That is disk/IO throughput, not RAM bandwidth.
2. The CPU then does the MoE math: 480 expert GEMVs per token (10 experts x 48 layers), plus the dense
   GEMVs of 2.85 GiB. Realistic CPU ceiling for that mix is ~10-20 t/s, so the CPU is not the wall yet,
   but it is not free either.
3. The best measured config (4.2) is exactly the one that put the dense set on devices and left the
   experts in host memory. It confirms the direction: **dense residency is the lever that is available
   today, expert residency is the lever that has not been tried.**
4. The `-n 4` generation in the current sweep makes every TG column noisy (4 tokens, page cache state
   from the 20k prefill included). Section 5 changes that.

## 3. What Vulkan0 (the 6 GB 4050) can and cannot do

Can:

- hold the entire dense set: 2.85 GiB + KV (580 MiB at 85k ctx, q4_1) + compute buffers, against ~5.0 GiB
  free at idle. Reads them at VRAM bandwidth instead of UMA/DRAM bandwidth: ~15 ms/token instead of
  ~53 ms/token for the dense pass.
- be the only device that can lift the dense term; the iGPU cannot (its "VRAM" is the same DRAM), CUDA0
  is the same silicon but goes through the slower cross-backend host copy path for the prefill
  (`tasks.md`, "CUDA0 as pp device: closed").

Cannot:

- hold the expert set (39.9 GiB), at any split. 6 GB of VRAM cannot be made to hold 40 GiB; the
  oversubscription that made the 40 GB incident possible is paging, not capacity.
- help the expert term in any way, since that term is bound by residency, not by compute.

## 4. Candidate configs, in the order worth testing

Fit arithmetic for the iGPU side: 40888 MiB experts + 580 MiB KV (85k) + ~750 MiB compute (ub 1024)
= ~42.2 GiB against 46508 MiB free. Fits with ~3 GiB of headroom; at ub 2048 the compute buffer grows
to ~1.5 GiB, still inside. PLE and `token_embd` stay host-side automatically (lazy flag, non-layer
tensor), so they do not compete for VRAM.

T1 - expert residency on the iGPU (untested, biggest single change)
```
-dev Vulkan1 -ngl 99
```
All 42.75 GiB of layer weights are staged into Vulkan1's UMA device buffer, so both the dense and the
sparse expert reads become resident DRAM reads and the cold-page term disappears. MUL_MAT_ID (the MoE
op) is supported on Vulkan for every quant type in this model (`ggml-vulkan.cpp:19532` switch lists
iq2_xs, iq2_s, iq2_xxs, iq3_s, iq4_nl, q2_0), so the MoE runs on the 760M, not the CPU.
Expect the dense term to still be UMA-bound: ceiling ~15 t/s, realistically 6-12.
Verify: `load_tensors: Vulkan1 model buffer size = ~43800 MiB`.

T2 - T1 + dense on the dGPU (the shape the user asked for, done by tensor class)
```
-dev Vulkan0,Vulkan1 -ts 0,100 -ot "<dense regex>=Vulkan0"
```
Layer assignment stays on the iGPU (KV, expert weights, PLE), the 2.85 GiB always-read dense set sits in
the 4050's VRAM. Per token this removes ~38 ms of the ~65 ms UMA-bound budget, at the cost of ~96 small
cross-PCIe activation copies per token (~5-24 KB each) and their syncs.
Dense regex (search match, full names like `blk.12.attn_qkv.weight`):
`blk\.[0-9]+\.(attn_|ssm_|hc_|ple_|ffn_(gate_inp|gate_shexp|up_shexp|down_shexp))`, plus
`output_hc_`, plus the 2 MiB `ffn_gate_inp`. Do NOT put `per_layer_token_embd` or `token_embd` here.
Expect: T1 plus ~1.3x, realistically 8-15 t/s if the Vulkan cross-device splits are cheap.
Fallback if V1 does not fit: `--n-cpu-moe N` (N = 2, 4, 6) moves the first N layers' experts to host and
frees 1.7 GiB per 2 layers.

T3 - dense on the dGPU + experts in host page cache (the cheap completion of s15cm)
```
-dev Vulkan0 -ts 100 --lazy-mode on --cpu-moe
```
Same as s15cm but with all 2.85 GiB of dense on Vulkan0 instead of split with Vulkan1 (s15cm put only
2826 MiB on V0 and 838 on V1). No residency for the experts, so it keeps the page-cache term, but it is
one run and it tells how much of the 4.2 was dense-residency vs noise.

T4 - T3 with CUDA0 instead of Vulkan0
```
-dev CUDA0 -ts 100 --cpu-moe
```
The user's point that CUDA is the usual backend deserves one data point: it isolates whether the dense
GEMVs are faster on CUDA than on Vulkan for this model (gemma smoke says CUDA0 prefill is 2.4x Vulkan0,
but that is prefill, and the prefill path is where CUDA loses for host weights).

T5 - CUDA0 dense + Vulkan1 expert residency (if T2 works and CUDA dense wins in T4)
```
-dev CUDA0,Vulkan1 -ts 0,100 -ot "<dense regex>=CUDA0"
```Not worth running: anything with a whole-layer `-ts` share on the 4050 above ~12% (it buys expert
  tensors at the price of dense residency), and `--pp-dev` for this model (`test_v1sc.md` section 3).

### 4c. Final matrix (protocol: 5k prompt, n 128, ctx 20000, b/ub 1024, fa on, q4_1 KV)

| tag | config | V0 buffer | V1 buffer | splits | prompt | **gen** |
|---|---|---|---|---|---|---|
| daily CUDA0 | `-dev CUDA0 --lazy-mode on` | 43 GB requested on a 6 GB card | - | - | 25.8 (20k) | 1.8 |
| t1 | `-dev Vulkan1 -ngl 99` | - | 44589 MiB | 2 | 75.0 | 5.6 |
| t4 | t1 + `--n-cpu-moe 8` | - | 38615 MiB (+6767 MiB host experts) | 26 / 18 | 56.3 | 4.8 |
| t2 | t1 + dense moved to V0 by tensor class | 3210 MiB | 41379 MiB | **509** | 75.1 | 2.3 |
| **t5** | `-dev Vulkan1,Vulkan0 -ts 90,10 -ngl 99` | **3486 MiB** (layers 43-47) | 41104 MiB | **3** | **84.7** | **7.2** |
| **t6** | t5 at ctx 85000 with the preset's `load-mode none` + `fitt 800MB` | 3486 MiB | 41104 MiB | 4 | 69.1 | **7.3** |

Readings:

1. **t5 wins on both axes**, and it is the only layout that uses the 4050 correctly. It gives the 4050
   whole layers (weights *and* experts), so the graph keeps 3 splits instead of 509, and 10% of the
   per-token FLOPs land on a device with ~10x the iGPU's throughput. t1 -> t5 is +29% TG and +13% PP.
   Against the config the machine runs today it is 4x TG (7.2 vs 1.8).
2. Adding a device by **tensor class** is the failure mode (t2, 2.3): every layer then holds a boundary,
   and each boundary is a backend switch plus a wait, ~500 of them per token.
3. Moving expert *compute* to the CPU (t4) does not pay on this box: the CPU side is slower than the
   760M even when the iGPU is the bottleneck, and it adds 18 decode splits. It is still the only row
   that reduces the unpageable footprint (6.0 GiB back to reclaimable file pages), so it is the choice
   when RAM pressure matters more than the last 14% of TG.
4. Prefill is not a problem in any of these (75-85 t/s at a 5k prompt, the iGPU-only host-weight configs
   were the ones at 118-194 because they streamed weights differently). t5 improved PP too.

Recommended config for this model: `-dev Vulkan1,Vulkan0 -ts 90,10 -ngl 99` (iGPU holds the model, the
4050 owns the last ~5 layers). Load + reserve is ~3-5 min, commit ~42 GB, single-stream TG 7.2-7.3 t/s.
The 4050 share can be pushed until its 5 GiB are full (~7 layers, `-ts 85,15` is the next step and may
OOM, `-ts 87,13` is the safe one).

### t6: preset-ready, verified at ctx 85000

Same layout, but with the preset's `[*]` settings: `--load-mode none -fitt 800MB -c 85000`. Loads clean,
TG 7.3 (t5's 7.2 reproduced), PP 69.1 (below t5's 84.7, the bigger KV allocation and the non-mmap load
path), graph splits 4. Placement: V0 3486 MiB weights + 71 MiB KV, V1 41104 MiB weights + 787 MiB KV,
`Vulkan1_Host` 260 MiB (token_embd), `CPU` 27465 MiB (PLE).

Two things this run settles for the preset:

1. `load-mode = none` does NOT force the 27.5 GiB PLE table into RAM - a lazy tensor is mapped whatever
the load mode (`llama-model-loader.cpp:1404`, `if (use_mmap || lazy.any())`). It shows up as a `CPU`
model buffer of 27465.95 MiB and stays sparse. So the preset's `load-mode = none` can stay.
2. The `--fit` projection before the load over-predicts the 4050: it printed
`Vulkan0 ... 6098 used, -1026 free vs. target of 800` and `cannot meet free memory targets ... need to
use 967 MiB less`, then aborted because `-ngl` was set by the user. The real allocation was 3486 MiB of
weights and the run never came close to OOM. Do not drop the V0 share just because of that projection.

Preset lines for this model:

```
[Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-V1+V0]
model                = IQ3_XXS\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
device               = Vulkan1,Vulkan0
ts                   = 90,10
ngl                  = 99
```

Leave `lazy-mode` unset (it would push fit off the `-ts` split and put the weights back in host memory),
and leave `[*]`'s `load-mode = none` and `fitt = 800MB` as they are.

### What the t1/t2 result changes

With t1 (5.6) the model is compute-bound on the 760M, and t2 shows the 4050 cannot be added per tensor.
The two rows left that attack the remaining wall:

- t4 `-dev Vulkan1 -ngl 99 --n-cpu-moe 8`: frees 8 x 851 MiB = 6.8 GiB of the unpageable Vulkan1
  buffer (those experts become file-backed and reclaimable) and moves 17% of the MoE work onto the
  idle CPU cores, i.e. it tests CPU/GPU overlap instead of fighting for the same 4 CUs. It is also the
  row that helps the RAM pressure.
- t5 `-dev Vulkan1,Vulkan0 -ts 90,10 -ngl 99`: layers 43-47 with their experts fully resident on the
  4050 (~4.25 GiB of its 5 GiB free), so ~10% of the FLOPs run on the fast GPU with ~2 split
  boundaries instead of 509. Modest, but it is the correct way to use the dGPU per the rule above.

### RAM cost vs the config the machine runs today

The Vulkan1 buffer is 44589 MiB of device memory, and a Vulkan device buffer on a UMA part is committed
system RAM that Windows cannot page out (observed: with the run loaded, free physical RAM 0.4 GB while
the pagefile stayed at 0.9 GB of 44 GB - nothing was paged, everything was trimmed).

The daily CUDA0 config is not cheaper: `-dev CUDA0 --lazy-mode on` asks for a 43 GB CUDA0 model buffer,
which on Windows is backed by shared system RAM for everything above the card's 6 GB. So the UMA layout
costs about the same RAM as what is already being run, at 3x the TG (5.6 vs 1.8). The RAM cost is a
property of serving a 70 GB model on this machine, not of the Vulkan backend.

## 4b. Results so far (protocol: 5k prompt, n 128, ctx 20000, ub 1024)

| tag | config | weights | splits | prompt | **gen** |
|---|---|---|---|---|---|
| t1 | `-dev Vulkan1 -ngl 99` | Vulkan1 buffer 44589 MiB (UMA), PLE 27465 MiB host, token_embd 260 host | **2** | 75.0 | **5.6** |
| t2 | t1 + `-dev Vulkan0,Vulkan1 -ts 0,100 -ot "<dense>=Vulkan0"` | V0 3210 MiB dense + V1 41379 MiB experts | **509** | 75.1 | 2.3 |
| t0 | `-dev Vulkan0 -ts 100 --lazy-mode on --cpu-moe` | dense 3664 MiB on V0, experts host | 145 | 2.4 (42 tok in 17 s) | dropped |

### t2 failed, and the split count says why

The dense-on-the-4050 idea loses 2.4x (5.6 -> 2.3) even though it moves 2/3 of the per-token FLOPs to a
gPU with ~10x the throughput. The reason is graph splits: t1 keeps the whole graph on one device
(**2 splits**), t2 interleaves tensor classes inside every layer (**509 splits**). Each split boundary
costs a backend switch, a submit and a wait; ~500 of them per token is ~0.4-0.5 s, which is exactly
the 2.3 t/s measured. Prefill is unaffected (75.0 vs 75.1) because prefill batches amortize the
boundaries.

Rule for this model: **a device can only be added at layer granularity, never by tensor class.** The
4050 is worth exactly the layers it can host entirely (its VRAM holds ~5 of 48 layers including their
experts, ~4.25 GiB), with one or two split boundaries, not 509.

T1 loads: all 49 layers staged into one 44589.69 MiB Vulkan1 buffer, private commit 44.8 GB, no
allocation error, and it runs. Against the old table that is TG 5.6 vs 4.2 best (s15cm) and 1.8 for the
CUDA0 lazy config the machine currently runs - and 5.6 was measured with a *worse* protocol (n=128 after
a 5k prompt instead of n=4 after a 20k prompt).

### Revised mechanism: t1 is compute-bound, not bandwidth-bound

3.9 GiB of weights per token at 5.6 t/s is only ~5.6 GB/s of effective DRAM traffic, an order of
magnitude under what the UMA can deliver. So expert+weight residency removed the disk/page-cache term
(that is what the jump from 1.5-4.2 to 5.6 is), and the new wall is the 760M's own math:
~13 GFLOP/token (2.85 GiB dense + 0.78 GiB of active experts) at 179 ms/token is ~74 GFLOPS effective,
which is about what 4 RDNA3 CUs do on quantized GEMV with dequant overhead.

Consequence for the remaining rows: if t1 is compute-bound on the iGPU, the dense set is not just
2.85 GiB of traffic, it is ~2/3 of the per-token FLOPs. Moving it to the 4050 (t2) attacks the wall
directly, so the expected gain is larger than the ~1.3x the traffic argument gives. The 4050 also has
~10x the quantized GEMV throughput of the 760M.

Also from t1: prefill on the iGPU runs 124 -> 95 t/s over the 4814-token prompt (KV growth), and the
run needs ~4.5 min of load+reserve before the first token, of which ~3 min is after the model buffer
lines (KV allocation and graph reserve with 43.5 GiB committed). That load cost is the price of UMA
residency and it is paid once per server start.

## 5. Measurement protocol for TG

The current sweep is a PP instrument: `-f prompt_20000.txt -n 4`. At 20k tokens of prefill it costs
7-13 min per point and the TG column is 4 tokens wide. For TG use:

```
-c 20000 -b 1024 -ub 1024 -fa on -ctk q4_1 -ctv q4_1 -t 6 -tb 6 \
  -n 128 -st -f prompts_perf/prompt_5000.txt
```
~1 min per point (5k prefill + 128 decode tokens), TG read from the `[ Prompt: X t/s | Generation: Y t/s ]`
line. Always read the placement back from the log (`model buffer size`, `KV buffer size`, `graph splits`)
because for this model the placement IS the experiment.

Risks to watch while these run, in order:

1. Load-time VRAM fit on Vulkan1 (42.2 GiB against 46.5 GiB free). If a run aborts at load, drop to
   `--n-cpu-moe 2/4/6` and note the number.
2. WDDM oversubscription again: any config with `-ngc`/`--lazy-mode on` and a user `-ts` on the 4050 can
   request tens of GB. The T2/T5 rows use `-ts 0,100`, which puts no layers on the dGPU, so they cannot
   repeat the incident; keep it that way.
3. RAM: T1/T2 hold 42 GiB of weights in UMA plus a 27 GiB PLE file being paged through. Keep the machine
   otherwise idle and watch commit charge during load.
