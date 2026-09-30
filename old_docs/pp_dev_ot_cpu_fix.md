# Why `--pp-dev` + `-ot ".*=CPU"` is 3x slower at TG, and what to change

Question: an agent added an uncommitted change to `common/common.cpp` so that `--pp-dev Vulkan0`
+ `-dev Vulkan1,Vulkan0` + `-ot ".*=CPU"` keeps some weights on Vulkan0 (V0). TG is ~3x worse than
before. What is wrong.

Short answer: the uncommitted change is not the cause. The cause is `--load-mode mmap` in the test
commands (it silently discards the `-ot ".*=CPU"` placement), plus `-dev Vulkan0,Vulkan1` in phase B
(the override then resolves to the 4050's host buft, which decode cannot read in place). The
uncommitted change should be reverted: an isolated A/B shows it costs TG and buys ~3% PP.

Against `sweep_27b_ud_pp_cpu_old` (pre-fix) the change is also not a 3x regression. Worst rows are a10
(TG 1.3 -> 1.0, -23%) and b15 (PP 103.0 -> 47.9, -53%); the 3x only appears when a pp-dev row is
compared with the non-pp-dev baseline (3.5-4.9 t/s), and that gap is the mmap effect of section 2. See
the full pre/post table in section 3b.

## 1. The uncommitted change

```
common/common.cpp, common_model_params_to_llama():
-    if (params.pp_dev != nullptr && params.pp_dev_resident) {
-        mparams.pp_dev_resident = params.pp_dev;
+    if (params.pp_dev != nullptr) {
+        if (params.pp_dev_resident || !params.tensor_buft_overrides.empty()) {
+            mparams.pp_dev_resident = params.pp_dev;
+        }
     }
```

Effect: any `-ot` now switches on `llama_model::load_tensors`'s resident-weight path
(`src/llama-model.cpp:1525`), which prepends `blk\.<il>\.` overrides for every layer that
`tensor_split` assigned to the pp device, so those weights stay in the pp device's own buft instead of
the host buft that `-ot ".*=CPU"` picks.

It does work mechanically. From `sweep_27b_ud_pp_cpu/a10.log`:
`keeping the weights of 5 layers in Vulkan0`, `Vulkan0 model buffer size = 1389.41 MiB`.

## 2. Where the 3x actually comes from

Both sweeps pass `--load-mode mmap` (`test_27b_ud_split_pp.bat`, `test_27b_ud_pp_cpu_pb.bat`, COMMON
line). With mmap on, `llama_model_loader::buft_for_tensor` rewrites any host buft to the CPU device
buft (`src/llama-model-loader.cpp:1305`):

```cpp
// avoid using a host buffer when using mmap
if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    buft = ggml_backend_dev_buffer_type(cpu_dev);
}
```

So `-ot ".*=CPU"` never reaches the per-device host buft. Every log in both sweep dirs shows it:

```
W llama_model_loader: tensor overrides to CPU are used with mmap enabled - consider using --load-mode none
D done_getting_tensors: tensor 'token_embd.weight' (and 865 others) cannot be used with preferred buffer type Vulkan1_Host, using CPU instead
I load_tensors: Vulkan1_Host model buffer size =     0.00 MiB
I load_tensors:   CPU_Mapped model buffer size = 16735.25 MiB
I sched_reserve: graph splits = 1187 (with bs=1024), 689 (with bs=1)
```

689 splits at bs=1 means the decode graph crosses back and forth for every op: the weight MUL_MATs sit
on the CPU (weights in `CPU_Mapped`, i.e. the mmap page cache) and everything else stays on the GPUs.
At decode, `ggml_backend_offload_op` is false anyway (`GGML_OP_OFFLOAD_MIN_BATCH`, 32), so nothing is
copied to a GPU and the CPU computes the model. TG is then bound by host RAM bandwidth: 16.7 GiB per
token, about 1 t/s. That is the whole regression.

`--load-mode none` keeps the override on the per-device host buft. Re-run of a00/a10 with the same
command, only load mode changed (`prompts_perf/vlm_*.log`):

| run | dev / ts | weights | bs=1 splits | PP t/s | TG t/s | draft acc |
|---|---|---|---|---|---|---|
| a00 sweep (mmap) | V1,V0 100,0 | CPU_Mapped 16735 | 689 | 108.4 | 1.0 | - |
| a10 sweep (mmap) | V1,V0 90,10 | CPU_Mapped 16735 | 689 | 116.9 | 1.3 | - |
| vlm_a00 (none) | V1,V0 100,0 | Vulkan1_Host 16735 | 97 | 97.0 | 2.6 | 0.36 |
| vlm_a10 (none) | V1,V0 90,10 | Vulkan0 1389 + Vulkan1_Host 15346 | 93 | 105.0 | 3.5 | 0.50 |
| vlm_a10nr (none, pre-fix binary) | V1,V0 90,10 | Vulkan1_Host 16735 (override wins) | 134 | 102.2 | 4.0 | 0.78 |
| vlm_a15 (none) | V1,V0 85,15 | Vulkan0 2215 + Vulkan1_Host 14520 | 89 | 46.2 | 3.8 | 0.53 |
| vlm_b05 (none) | V0,V1 5,95 | Vulkan0 872 + Vulkan_Host 15863 | 647 | 112.7 | 1.0 | 0.46 |
| vlm_b15 (none) | V0,V1 15,85 | Vulkan0 2176 + Vulkan_Host 14560 | 582 | 48.4 | 1.1 | 0.50 |

Protocol: same as the sweep but `-n 32` (ctx 85000, b/ub 1024, fa on, q4_1 KV, 4814 token prompt,
MTP n-max 3, `--pp-dev Vulkan0 -ot ".*=CPU"`).

Two more findings in that table:

- Device order decides everything. With `-dev Vulkan0,Vulkan1`, `-ot ".*=CPU"` resolves to
  `Vulkan_Host`, the 4050's host buft. The 4050 is discrete, so decode cannot read it in place and
  bs=1 splits stay at 582-647 (same shape as the CPU case): TG 1.0-1.1 even with the right load mode.
  Phase B of the sweep is dead on arrival for TG; only `-dev Vulkan1,...` (UMA iGPU first) gives
  zero-copy host weights.
- The `ts 85,15` PP cliff (105.0 -> 46.2 for +3 resident layers, no allocation error, identical
  compute buffers) matches the note in `handoff.md` 8.9: PP collapses once the V0 footprint passes
  about 4.5 GB. `ts 90,10` is the practical ceiling for this model at ctx 85000.

Caveat: `-n 32` + MTP makes the TG column noisy, the observed draft acceptance ranges 0.36 to 0.78
between runs. The splits and the buffer placements are the hard evidence; re-measure TG with the
`x00`/`x15` (no-spec) rows or a larger `-n`.

## 3. Is the uncommitted change worth keeping

No. Isolated A/B, identical command and identical `-ts 90,10`, both `--load-mode none`, only the
`common/common.cpp` change toggled (pre-fix binary built by stashing the change):

| | weights | bs=1 splits | PP t/s | TG t/s |
|---|---|---|---|---|
| with the change (`vlm_a10`) | Vulkan0 1389 + Vulkan1_Host 15346 | 93 | 105.0 | 3.5 |
| without (`vlm_a10nr`) | Vulkan1_Host 16735 (override wins) | 134 | 102.2 | 4.0 |

Moving those 5 layers into V0's VRAM buys ~3% PP and costs TG, which is the same verdict as the D-series
in `handoff.md` 8.9 finding 3 ("weight residency on Vulkan0 is not needed"). TG costs the acceptance
difference too, so treat 3.5 vs 4.0 as "not a win" rather than a measured 12% loss.

The change also should not be kept for a design reason: it makes weight placement, VRAM use and graph
shape depend on whether an unrelated `-ot` is present. A user writing `-ot "ffn.*=CPU"` for some other
reason silently gets pp-dev residency. `--pp-dev-resident` already exists as the explicit switch.

Why it looked catastrophic in the mmap logs: with `--load-mode mmap` the resident bookkeeping is
inconsistent. Order B counts the model twice, `CPU_Mapped 16735.25 + Vulkan0 2175.67` for a 16.35 GiB
model, and PP halves (b15: 103.0 -> 47.9). With `--load-mode none` the accounting is exact
(a10: 1389.41 + 15345.84 = 16735.25), so that is a reporting artifact of the mmap CPU_Mapped
reservation, not a real second allocation.

## 3b. Pre-fix vs post-fix sweep, row by row

`sweep_27b_ud_pp_cpu_old` is the pre-fix binary (no `Vulkan0 model buffer size` line anywhere, all
weights in `CPU_Mapped`). `sweep_27b_ud_pp_cpu` is the same bats after the rebuild with the change.
Both use `--load-mode mmap`, so the CPU-decode regime of section 2 applies to every row.

| tag | dev / ts | pre-fix PP / TG | post-fix PP / TG | resident layers | resident MiB |
|---|---|---|---|---|---|
| a00 | V1,V0 100,0 | 108.4 / 1.0 | 102.8 / 0.9 | 0 | - |
| a05 | V1,V0 95,5 | 111.5 / 1.2 | 103.4 / 1.0 | 2 | 619.24 |
| a10 | V1,V0 90,10 | 116.9 / 1.3 | 103.7 / 1.0 | 5 | 1389.41 |
| b05 | V0,V1 5,95 | 104.2 / 0.8 | 105.4 / 1.0 | 4 | 871.85 |
| b10 | V0,V1 10,90 | 109.4 / 1.1 | 111.2 / 1.1 | 7 | 1555.65 |
| b15 | V0,V1 15,85 | 103.0 / 0.9 | 47.9 / 1.1 | 10 | 2175.67 |
| x00 (no spec) | V1,V0 100,0 | 94.3 / 1.3 | 91.6 / 1.3 | 0 | - |
| x15 (no spec) | V1,V0 85,15 | 94.7 / 1.1 | 96.9 / 1.1 | 8 | 1880.27 |

Worst rows: a10 PP -11% / TG -23%, b15 PP -53% / TG +22%. Nowhere near 3x. bs=1 splits move the same
way in both directions: a10 689 -> 648, b15 689 -> 582, a05/x15 689 -> 681/616. The pre-fix rows also
have no resident layers at all, so a pre-fix a10 is not "the same config minus residency" - it is
"all weights on the host buft" (see the isolation run in section 3 for the clean comparison).

Draft acceptance per row (`-n 128`, so these are more stable than the `-n 32` runs of section 2):
a00 0.376 -> 0.459, a05 0.433 -> 0.527, a10 0.474 -> 0.412, b05 0.316 -> 0.477, b10 0.484 -> 0.401,
b15 0.384 -> 0.487, x00/x15 no spec. With acceptance moving 0.32 to 0.53 between runs of the same
config, the TG column of these two sweeps cannot resolve anything smaller than roughly 20%.

## 4. What to change

1. Revert `common/common.cpp`. Keep residency behind the explicit `--pp-dev-resident` flag.
2. In the sweep bats, `--load-mode mmap` -> `--load-mode none` on every run that uses
   `-ot ".*=CPU"`. This is the branch's own acceptance criterion (`handoff.md` 16.8, section 9.0):
   `-dev Vulkan1 -ngl 99 -ot ".*=CPU" --load-mode none --pp-dev Vulkan0` -> about 131 t/s prefill at
   ub 3072 with zero-copy decode.
3. Keep `-dev Vulkan1,Vulkan0` (iGPU first). Phase B (`-dev Vulkan0,Vulkan1`) is a control for the
   wrong order, not a candidate.
4. Re-run the sweep after 1 and 2, with the `-n 128` protocol on the `x00`/`x15` rows so TG is not
   dominated by MTP acceptance.
5. For this 27B at ctx 85000 with pp-dev, expect PP about 100-110 and TG about 3.5-4.0 with weights in
   `Vulkan1_Host`. The non-pp-dev baseline is PP about 50 / TG about 4.7, so the trade is
   "2x prefill for about 20% decode".

## 5. Verification logs

`prompts_perf/vlm_a00.log`, `vlm_a10.log`, `vlm_a15.log`, `vlm_a10nr.log`, `vlm_b05.log`,
`vlm_b15.log`. Command template:

```
llama-cli -m Qwen3.8-27B-UD-Q4_K_XL.gguf --load-mode none -fa on -ctk q4_1 -ctv q4_1 -t 6 -tb 6 \
  -np 1 --cache-ram 0 -n 32 -st -c 85000 -b 1024 -ub 1024 -f prompt_5000.txt \
  --pp-dev Vulkan0 -ot ".*=CPU" --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-type-k q4_1 --spec-draft-type-v q4_1 --spec-draft-device Vulkan0 -lv 5 \
  -dev Vulkan1,Vulkan0 -ts 90,10
```

`a10nr` used the same command with the `common/common.cpp` change stashed. The working tree and
`build/bin` were restored to the pre-test state afterwards.

## 6. Every TG >= 4 config in prompts_perf, and which rows are trustworthy

Search: all `prompts_perf/*/summary*.txt`, then the `eval time` line of each winning log to check that
the number is a real measurement and not a single lucky MTP step.

### Valid (n=128 generated tokens)

Qwen3.8-27B-UD-Q4_K_XL, `--load-mode mmap`, **no `-ot`, no `--pp-dev`** (weights in `Vulkan1`, 16053.22
MiB, plus `CPU_Mapped` 682.03 MiB, graph splits = 2), ctx 85000, b/ub 1024, 5k prompt, MTP n-max 3:

| sweep | tag | dev / ts | MTP dev | PP | TG | draft acc |
|---|---|---|---|---|---|---|
| sweep_27b_ud (Sep 28) | a05 | V1+V0 95,5 | V0 | 45.1 | 4.3 | 0.456 |
| | a15 | V1+V0 85,15 | V0 | 54.0 | 4.9 | 0.481 |
| | m15v1 | V1+V0 85,15 | V1 | 52.0 | 4.5 | - |
| | m15cpu | V1+V0 85,15 | none | 57.4 | 4.6 | - |
| | b05 | V0+V1 5,95 | V0 | 50.9 | 4.4 | 0.451 |
| | b10 | V0+V1 10,90 | V0 | 54.4 | 4.0 | 0.389 |
| sweep_27b_ud_again (Sep 29) | a00 | V1+V0 100,0 | V0 | 49.4 | 4.7 | 0.538 |
| | a05 | V1+V0 95,5 | V0 | 49.3 | 4.1 | 0.373 |
| | m15v1 | V1+V0 85,15 | V1 | 56.0 | 4.7 | - |
| | m15cpu | V1+V0 85,15 | none | 55.8 | 4.8 | - |
| | b10 | V0+V1 10,90 | V0 | 53.7 | 4.3 | 0.414 |
| | b15 | V0+V1 15,85 | V0 | 57.9 | 5.0 | 0.527 |

Flash-Next `IQ3_XXS` (different model, 70.6 GiB, `sweep_tg`, ctx 20000, n 128):

| tag | config | PP | TG |
|---|---|---|---|
| t1 | `-dev Vulkan1` | 75.0 | 5.6 |
| t4 | t1 + `--n-cpu-moe 8` | 56.3 | 4.8 |
| t5 | `-dev Vulkan1,Vulkan0 -ts 90,10` | 84.7 | 7.2 |
| t6 | t5 at `-c 85000` + `--load-mode none` | 69.1 | 7.3 |

Reproducibility: the same configs on the two days spread about 20%. Only `m15cpu` (4.6 / 4.8),
`m15v1` (4.5 / 4.7) and `b10` (4.0 / 4.3) clear 4 twice. `a15` gives 4.9 / 3.9 and `b15` 3.9 / 5.0,
so single rows in that band are not evidence.

Note what every valid >= 4 row has in common: no `-ot`, no `--pp-dev`, so the weights are in
`Vulkan1` (the iGPU's own buffer, splits = 2) and PP is 45-58 t/s. The pp-dev configs give up part of
that decode for 2-2.5x the prefill.

### Invalid (do not cite these)

`sweep_ts` and `sweep_mtp` look like the fastest 27B rows in the tree, and they are an artifact. Every
one of them generated only 4 tokens in one fully accepted MTP step:

```
eval time =     619.79 ms /     4 tokens (  206.60 ms per token,     4.84 tokens per second)
draft acceptance = 1.00000 (    2 accepted /     2 generated), mean len =  3.00
```

Affected rows: `sweep_ts` A0 4.8, A15r 5.4, A15r_m1 5.5, A15n 4.9, A20r_m1 4.5 (`--pp-dev-resident`,
ctx 20224, b/ub 1536); `sweep_mtp` 5.4 / 5.2 / 5.1 / 4.9 / 4.8 / 4.3 (`-ot "*=CPU"` + `--pp-dev`,
weights in `Vulkan1_Host` 16735.25 MiB); `sweep_v1s` s15cm 4.2 (705.93 ms / 4 tokens). The `-n` used
was 4, and the first draft was accepted 2/2, so the reported rate is one step, not a steady state.

That matters because those rows are the only place where `--pp-dev-resident` ever looked like a TG
win (`A15r` 5.4 vs `A15n` 4.9). With `-n 128` the same family of configs measures 3.5-4.0 (section 2
and the isolation run in section 3), and the resident flag costs a little on TG. Re-run `sweep_ts`
with `-n 128` before believing anything in that table.
