# Qwen3.8-27B-UD-Q4_K_XL: split + MTP placement sweep

Script: `test_27b_ud_split.bat` (dry run with `set "DRYRUN=1"`)
Logs/summary: `prompts_perf/sweep_27b_ud/`
Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17559178144 B = 16.35 GiB, arch `qwen35`
MTP: the file carries its own MTP layers (`nextn.*` tensors, `qwen35.nextn_predict_layers`), so
`--spec-type draft-mtp` runs off the same file. The draft instance loads only the MTP layers
(`load_mtp` in the model loader), so its footprint is small and it can sit on the 4050 next to a
main-model share without competing for VRAM.

## What the sweep varies

| axis | why it is in the sweep |
|---|---|
| `-ts` V1:V0 share, 0/5/10/15/20/25 % | how much of the 16.35 GiB and its KV goes to the 4050. 5.0 GiB free on V0 caps this near 25 % once the V0 KV share and its compute buffers are counted, so 25 % is the fit probe |
| `-dev` order, `Vulkan1,Vulkan0` and `Vulkan0,Vulkan1` | order sets the main device, and with it where the scheduler starts, where the output layer and the logits live, and which device owns the first host buffer type. b-rows are the same allocations as the a-rows with the orders swapped |
| `--spec-draft-device` Vulkan0 / Vulkan1 / none | where the MTP head runs. Vulkan0 is the 4050, Vulkan1 is the iGPU, `none` leaves the draft on the CPU (the flag takes a device list, and a single `none` means "do not offload") |
| no-spec controls | prices MTP itself: x00 vs a00, x15 vs a15 |

13 rows, ~3 min each (5k prompt, n 128, ctx 85000, b/ub 1024, q4_1 KV, fa on, MTP n-max 3).
`-lv 5` is required, the `draft acceptance = X (a / b), mean len = Y` line only prints at that level
and it is the metric that says whether MTP is actually paying off.

## Rows

| tag | dev | ts | mtp dev |
|---|---|---|---|
| a00 | Vulkan1,Vulkan0 | 100,0 | Vulkan0 |
| a05 | Vulkan1,Vulkan0 | 95,5 | Vulkan0 |
| a10 | Vulkan1,Vulkan0 | 90,10 | Vulkan0 |
| a15 | Vulkan1,Vulkan0 | 85,15 | Vulkan0 |
| a20 | Vulkan1,Vulkan0 | 80,20 | Vulkan0 |
| a25 | Vulkan1,Vulkan0 | 75,25 | Vulkan0 |
| m15v1 | Vulkan1,Vulkan0 | 85,15 | Vulkan1 |
| m15cpu | Vulkan1,Vulkan0 | 85,15 | none |
| b05 | Vulkan0,Vulkan1 | 5,95 | Vulkan0 |
| b10 | Vulkan0,Vulkan1 | 10,90 | Vulkan0 |
| b15 | Vulkan0,Vulkan1 | 15,85 | Vulkan0 |
| x00 | Vulkan1,Vulkan0 | 100,0 | no spec at all |
| x15 | Vulkan1,Vulkan0 | 85,15 | no spec at all |

## What to look for, and what earlier data says

- The 27B MXFP4 sweep (`prompts_perf/sweep_ts/summary.txt`) put the best decode at
  `Vulkan1+Vulkan0 ts 85+15` with the MTP head on either device (142.4/5.4 on Vulkan0, 142.4/5.5 on
  Vulkan1), and 80+20 collapsed (72.6/3.1) while 75+25 with the MTP head on Vulkan1 held at
  55.2/3.2. That sweep used 1536/1536 and the `--pp-dev-resident` variants, so the numbers are not
  directly comparable, but the shape says the useful window is 10-20 % and that pushing past it costs.
- The Flash-Next finding applies here: a device must be added at **layer** granularity. `-ts` does
  that by construction, so these rows are safe; the trap would be mixing `-ot` tensor-class overrides
  into the split (that cost 2.4x on the 47 GB model, 509 graph splits instead of 3).
- Watch `graph splits` in each log: a-rows and b-rows with the same ts should keep the same split
  count, so any difference between them is the order effect and nothing else.
- Watch `draft acceptance` and `mean len`: MTP only pays if acceptance stays high. On KAT-Coder the
  Q4_1 draft KV gave 0.64 acceptance and 2.92 mean len; if a row shows acceptance dropping, the MTP
  head has landed somewhere too slow for its own latency budget and the gain disappears.
- No `--pp-dev` in this sweep: prefill is accepted as-is, and `--pp-dev-resident` costs decode
  (it puts layers and their KV on the pp device, which adds decode splits).

## Results

Fill in as rows complete, `tag,dev,ts,mtp_dev,prompt_tps,gen_tps,draft_acc` is the CSV in
`prompts_perf/sweep_27b_ud/summary.txt`.
