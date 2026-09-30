# GPU attribution: where the 15 GB of shared RTX VRAM came from, and what V0 does during TG

Follow-up to `pp_dev_ot_cpu_fix.md`. Two questions: why did the RTX 4050 (V0) show 15+ GB of shared
VRAM, and why is V0 busy during TG at all.

## 1. The monitor

`prompts_perf/gpumon.ps1`, 1 Hz, writes `gpumon_<tag>.csv` next to itself:

```
powershell -NoProfile -ExecutionPolicy Bypass -File prompts_perf/gpumon.ps1 -Tag a10 -Seconds 300
```

Columns: `t_wall`, `elapsed_s`, `nv_util_pct`, `nv_ded_mib` (nvidia-smi memory.used),
`nv_shared_mib` and `nv_ded_adapter_mib` (the Windows counters Task Manager uses,
`\GPU Adapter Memory(*)\Shared Usage` / `Dedicated Usage`), plus `proc_ded_mib` / `proc_shared_mib`
summed over the pids of `-ProcName` (default `llama`). The RTX adapter instance (LUID) is detected at
start by matching adapter Dedicated Usage against nvidia-smi.

Alignment with the llama.cpp log: every llama line starts with `H.MM.SS.mmm` elapsed since process
start, and the csv has wall clock plus `elapsed_s` from the first sample. Start the monitor and the
run within a second of each other and the two time bases line up to the second.

Note on `GGML_CUDA_NO_PINNED`: it propagates fine from bash (`export GGML_CUDA_NO_PINNED=1` then
`cmd //c "echo %GGML_CUDA_NO_PINNED%"` prints 1), but it only affects the CUDA backend, two call sites
(`ggml-cuda.cu:1277` `ggml_cuda_host_malloc`, `:5023` `props->caps.host_buffer`). It cannot change a
Vulkan host buffer. Bat files are still the right place for it, since that is what the launchers do.

## 2. The 15+ GB is real, and it is the old build's host buffer

Same model, same flags, only the binary differs. `-dev Vulkan1 -ngl 99 -ot ".*=CPU" --load-mode none`,
27B UD-Q4_K_XL:

| binary | weights land in | peak RTX shared | peak RTX dedicated | PP | TG | draft acc |
|---|---|---|---|---|---|---|
| `D:\progs\llama-pp` (10879, no `--pp-dev`, no per-device host buft) | **`Vulkan_Host` 16735 MiB** | **17434 MiB** | 1747 MiB | 34.2 | 2.4 | 0.71 |
| `D:\progs\llama-cpp-pw` (11135, per-device host buft) | `Vulkan1_Host` 16735 MiB | **302 MiB** | 2978 MiB | 106.4 | 3.4 | 0.58 |

Logs: `prompts_perf/llamapp_ot_mon.log`, `pwpw_preset_mon.log`, csv `gpumon_llamapp_ot.csv`,
`gpumon_pw_preset.csv`.

Mechanism, all three pieces verified in this tree:

1. `-ot ".*=CPU"` resolves through `select_weight_buft` against `buft_list_cpu`, and that list offers
   exactly one host buft: the one of the FIRST device in `-dev` (`make_cpu_buft_list`,
   `src/llama-model.cpp:1052-1059`, `for (const auto & dev : devices) { ... break; }`).
2. On a build without the per-device host buft (this fork's `ecf903658`), that host buft is the global
   `Vulkan_Host`, whose allocator is pinned to **Vulkan device 0**:
   `ggml_vk_host_malloc(vk_instance.devices[0], size)` (`ggml-vulkan.cpp:17261`). On this machine
   Vulkan device 0 is the RTX 4050, so 16.4 GB of host-visible memory is allocated through the NVIDIA
   driver and WDDM counts it as the 4050's shared GPU memory. That is the 17 GB.
3. The scheduler then treats those weights as owned by Vulkan0 and runs their matmuls on the 4050 with
   the data in host RAM. So the RTX is busy in TG and every token pays PCIe. Hence TG 2.4 with a
   16.4 GB host buffer, versus TG 3.4-4.4 and 302 MiB of shared memory on a build with the per-device
   host buft.

Task Manager cross-check, from the same two csv files. During the `llama-pp` run the RTX row reads
16938 MiB shared at 0-5% utilization (peak 23%), while the Radeon is at 94% doing all the work: the
weights were allocated on the NVIDIA device but the scheduler still sends the weight ops to the first
GPU in the backend list, which is Vulkan1 because `-dev Vulkan1,...` comes first. So the iGPU computes
while reading weights out of an NVIDIA-owned host buffer, and the RTX just holds 17 GB and idles.
That is exactly the reported symptom (`RTX has 17 GB shared, RX only 3`). With the per-device host buft
(`llama-cpp-pw`) the same run shows 302 MiB shared on the RTX, RTX utilization up to 50% (it gets real
work: the `-ts 90,10` layers and the MTP draft), and TG 3.4-4.4 instead of 2.4.

Acceptance test, no tools needed: with `-ot "*=CPU" --load-mode none` the ~16.7-20 GB must appear in
the **Radeon 760M** row of Task Manager (its shared GPU memory), and the RTX must stay around 0.3 GB
shared / 3 GB dedicated. If the 17 GB is in the RTX row, the weights are on the wrong device and TG is
in the 1-2.4 t/s range.

The same symptom appears on a modern build if `-dev` lists `Vulkan0` first, because the host buft of
the first device is the one that gets picked (`make_cpu_buft_list`). Keep the iGPU first.

Do not use `llama-pp` with any `ot = .*=CPU` preset. That binary also lacks `--pp-dev` entirely
(`--help | grep pp-dev` = 0 matches), so the pp-dev entries in `D:\script\preset.ini`
(`[Qwen3.8-UD_Q4_XL-V1-V0-PP-MTP]`, `[Qwen3.8-27B-MXFP4-V1-V0-MTP]`) cannot work there either: both
have `ot = .*=CPU` active with `pp-dev` commented out, which is exactly the 17 GB / 2.4 t/s case.

Fix for the launcher: `llama-_server_test.bat` does not set `GGML_CUDA_NO_PINNED=1` while
`llama-_server.bat` does. It is not what causes the 17 GB (Vulkan, not CUDA), but it does change what
CUDA0 advertises (`caps.host_buffer`), so set it for consistency.

## 3. How much memory actually goes to VRAM

Per device, from `pwpw_preset_mon.log` (`-dev Vulkan1,Vulkan0 -ts 90,10 --pp-dev Vulkan0`), the config
of the new preset entry:

| device | buffer | MiB |
|---|---|---|
| Vulkan0 (4050, dedicated) | KV (5 layers) | 104.06 |
| | RS | 37.41 |
| | compute (main ctx) | 303.72 |
| | compute (MTP draft ctx) | 579.53 |
| | **total VRAM** | **1024.7** |
| Vulkan1 (iGPU, UMA) | KV | 1560.94 |
| | RS | 561.09 |
| | compute | 200.46 |
| | total | 2322.5 |
| Vulkan1_Host (host RAM, `-ot ".*=CPU"`) | model | 16735.25 |

With our build's resident auto-enable the 4050 also takes `Vulkan0 model buffer size = 1389.41` MiB,
which brings V0 to 2414 MiB and drops `Vulkan1_Host` to 15345.84. `vlm_a10.log` versus `vlm_a10nr.log`.

So the 4050 is assigned about 1.0 GB (2.4 GB with residency), never 15 GB. Anything near 15 GB is
mis-attributed host memory, which is the section 2 case.

For reference, the older mmap-based sweeps (`sweep_27b_ud_pp_cpu*`) put the same 16.7 GB in
`CPU_Mapped` instead, with V0 at 1024.7 MiB as well; the difference is the CPU page cache versus the
iGPU's host-visible buffer, and that is the whole 1.0 versus 2.6-4.4 t/s story in
`pp_dev_ot_cpu_fix.md` section 2.

## 4. Why V0 is busy during TG

Three separate reasons, all in the log above, none of them a bug:

1. `-ts 90,10` gives V0 the last 5 layers, so their attention, SSM and norm ops plus their KV slice
   (104 MiB) run on the 4050 on every token.
2. `--spec-draft-device Vulkan0` runs the whole MTP draft model on the 4050 every token. Its own
   context reserves the 579.53 MiB compute buffer, the largest single V0 allocation in the list.
3. Our build additionally keeps 1389.41 MiB of weights in V0 (the resident path).

If V0 activity during TG should be avoided, the levers are `-ts 100,0` (V0 owns no layers) and
`--spec-draft-device Vulkan1` or `none`. Both were measured in the two 27B sweeps; the trade is in
`pp_dev_ot_cpu_fix.md` section 6. `gpumon.ps1` now makes the effect of each visible per second.

## 5. What this does not explain yet

TG 3.4 versus 4.4 for two identical runs of the preset config on `llama-cpp-pw` (acceptance 0.58 versus
0.61, so it is not acceptance). That is either machine state (RAM pressure, file cache) or something
in the graph. The per-split tracer in `pp_tg_trace_plan.md` is the tool for it: R1 (TG budget per
token) plus R6 (idle fraction) on both runs, with `gpumon` running alongside.
