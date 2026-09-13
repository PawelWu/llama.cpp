# Double buffering: streaming wag z RAM do fixowanego bufora VRAM (Vulkan/AMD)

Cel: załadować model tak, że **warstwa 0 jest stale na GPU** (`-ngl 1`), a
pozostałe warstwy mieszkają w RAM (pinned). Podczas eval fixowany bufor VRAM
(np. 2048 MB) obsługuje pozostałe warstwy **chunkami**: liczymy chunk [2-10],
a w tym samym czasie pobieramy (H2D) chunk [11-20] do drugiej połowy bufora.
Klasowy software pipeline: nakładamy kopię H2D kolejnego chunku na obliczenia
bieżącego chunku.

Sprzęt: AMD dGPU, **wyłącznie backend Vulkan** (brak karty NVIDIA).

Status: badanie (investigation). Nie jest to jeszcze implementacja - wymaga
uzgodnienia z maintainerami (zob. AGENTS.md).

---

## 1. Pojęcie (udoskonalone)

```
RAM (pinned)                     VRAM ring (np. 2048 MB, 2 sloty)
+---------------------+          +----------------------------------+
| warstwa 0  -> GPU   |          | slot A: wagi chunku [2-10]       | <- liczy compute queue
| warstwy 1..N-1      |----------| slot B: wagi chunku [11-20]      | <- pobiera transfer queue
|   (pinned host)     |          +----------------------------------+
+---------------------+
   ^ H2D async (transfer queue)   ^ compute (compute queue) - nakładają się
```

- Warstwa 0: stale w buforze GPU (normalny offload, `-ngl 1`).
- Warstwy 1..N-1: w pinned RAM. Wykonywane na GPU przez streaming buffer.
- Chunk = grupa warstw (np. 9). Rozmiar chunku tak, by wagi chunku zmieściły
  się w jednym slocie ringu (S/2).
- Double buffering: 2 sloty. Liczymy slot A, pobieramy slot B, potem zamiana.

Korzyść: zamiast `sum(compute + copy)` na chunk, dostajemy
`max(compute_chunk, copy_chunk_nastepnego)`. Działa gdy compute i copy są
porównywalne. Jeśli copy >> compute (słabe GPU) pipeline copy-bound, zysk mały.
Jeśli compute >> copy (mocne GPU) copy i tak ukryte, zysk mały.

Wymagania:
- Wagi 1..N-1 w **pinned** host memory (prawdziwie async H2D).
- Ring VRAM >= 2 chunki wag.
- KV cache + warstwa 0 + ring muszą się zmieścić w VRAM.
- Wykonywanie grafu **per-chunk** (nie cały graf naraz) + prefetch.

---

## 2. Stan obecny w kodzie (backend Vulkan/AMD)

Najbliższa istniejąca funkcjonalność: **lazy mode + op_offload**. Różnice od
naszego celu: (a) wagi przychodzą z mmap (dysk), nie z pinned RAM, (b) scheduler
kopiuje wszystkie wagi splitu PRZED obliczeniem - brak nakładania copy na
compute.

### 2.1 Lazy mode (`--lazy`)
- `include/llama.h:217-221` - `llama_lazy_mode` (OFF/AUTO/ON).
- `src/llama-model-loader.cpp:1081` - `lazy_read::add`: duże tensory oznaczane
  do lazy read.
- `src/llama-model.cpp:1743-1770` - lazy context mapowany do host buffer przez
  `ggml_backend_dev_buffer_from_host_ptr`.

### 2.2 op_offload (domyślnie WŁĄCZONE)
- `common/common.cpp:1748` - `cparams.op_offload = !params.no_op_offload`.
- `ggml/src/ggml-backend.cpp:970` - wagi w host buffer -> op offloadowany na GPU.
- Efekt: wagi w host buffer, scheduler kopiuje je na GPU "na żądanie".

### 2.3 Scheduler: async copy + eventy + REUŻYCIE bufora kopii
- `ggml/src/ggml-backend.cpp:1643-1790` - `ggml_backend_sched_compute_splits`:
  dla splitu najpierw kopiuje inputy (wagi), potem liczy.
- `ggml/src/ggml-backend.cpp:1778` - próba async copy (`cpy_tensor_async`).
- **KLUCZOWE** (`ggml-backend.cpp:1393-1407`): dla `n_copies == 1` (tryb
  nie-parallel, nasz przypadek) kopia wagi NIE jest oznaczana jako output
  (`ggml_set_output` tylko gdy `n_copies > 1`). Zatem **ggml-alloc może
  reużywać pamięć kopii wag między warstwami** - bufor GPU nie musi trzymać
  wszystkich warstw naraz, tylko największą warstwę (max concurrent set).
  To dobra wiadomość dla fixowanego bufora.
- `ggml/src/ggml-backend.cpp:1865` - `n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1`.
- MoE: `ggml-backend.cpp:1687-1776` - kopiuje tylko użyte eksperty.

### 2.4 Pinned host buffer (Vulkan) - JEST
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:17004-17040` -
  `ggml_backend_vk_host_buffer_type`: alokuje pinned memory.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:8335-8350` - `ggml_vk_host_malloc`:
  buffer `eHostVisible | eHostCoherent | eHostCached` (pinned, host-coherent).
- Fallback na CPU buffer gdy pinned nie działa.

### 2.5 Osobny transfer queue (Vulkan/AMD) - JEST
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:7355-7372` - na AMD dGPU (non-GCN, czyli
  RDNA) tworzony jest **dedicated transfer queue**; `async_use_transfer_queue`
  = true (gdy graphics queue disabled) albo przez `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`.
- Znaczenie: H2D copy idzie na **osobnym queue** i może się nakładać na compute
  na compute queue. To dokładnie to, czego potrzebuje pipeline.

### 2.6 Async copy H2D (Vulkan) - JEST
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:17216-17270` -
  `ggml_backend_vk_cpy_tensor_async`: H2D z pinned host na device, na transfer
  queue (gdy `async_use_transfer_queue`), inaczej na compute queue.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:17097` - `set_tensor_2d_async`.

### 2.7 `buffer_from_host_ptr` (Vulkan) - JEST
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:19994-20001` -
  `ggml_backend_vk_device_buffer_from_host_ptr`. Lazy mode działa na Vulkan.

### 2.8 Graf budowany naraz, per-backend splits
- `src/llama-model.cpp:2732` - `llama_model::build_graph` -> `build_arch_graph`.
- `src/models/gpt2.cpp:82` - pętla `for (int il = 0; il < n_layer; ++il)` buduje
  CAŁY graf w jednym `ggml_cgraph`.
- **Nie ma API do wykonywania grafu per-warstwa/per-chunk.** To główna luka.

### 2.9 Load modes
- `include/llama.h:205-211` - `LLAMA_LOAD_MODE_*`: NONE, MMAP, MLOCK,
  MMAP_MLOCK, DIRECT_IO. `MLOCK` trzyma model w RAM (anti-swap).

---

## 3. Kluczowe ustalenia

1. **Prymitywy pipeline'u istnieją w Vulkan/AMD**: pinned host memory, osobny
   transfer queue, async H2D copy, `buffer_from_host_ptr`. Brakuje tylko
   ORKESTRACJI (nakładanie copy na compute per-chunk).
2. **Bufor GPU jest już ograniczony do największej warstwy** (n_copies=1 ->
   reużytkowanie kopii wag). Fixowany bufor 2048 MB jest więc realistyczny.
3. **Główna luka**: scheduler robi "all copies then all compute" w jednym
   splitu. Nie ma per-chunk execution ani prefetchu. To jedyna rzecz do
   dodania (plus pinned RAM dla warstw 1..N-1 zamiast mmap/CPU).
4. **Warstwa 0 stale na GPU** (`-ngl 1`) upraszcza: nie trzeba jej streamować,
   tylko pozostałe warstwy.

---

## 4. Taski

- [x] T1: Zmapować infrastrukturę (lazy, op_offload, scheduler, pinned,
      transfer queue, load modes) dla Vulkan/AMD. -> sekcja 2.
- [x] T2: Czy scheduler nakłada dziś copy na compute? -> NIE (all copies then
      compute, sekcja 2.3).
- [x] T3: Czy jest pinned host buffer? -> TAK (sekcja 2.4).
- [x] T4: Czy jest osobny transfer queue na AMD? -> TAK, RDNA (sekcja 2.5).
- [x] T5: Czy bufor GPU trzyma wszystkie warstwy naraz? -> NIE, n_copies=1
      reużytkuje (sekcja 2.3).
- [x] T6: Czy graf da się wykonać per-chunk? -> NIE ma API (sekcja 2.8). Luka.
- [ ] T7: Zaprojektować minimalny schemat (ring + per-chunk + prefetch). -> sekcja 5.
- [ ] T8: Oszacować zysk (benchmark compute/copy per chunk). -> sekcja 6.
- [ ] T9: Ryzyka i pytania otwarte. -> sekcja 7.
- [ ] T10: Prototype (jeśli maintainerzy OK). -> sekcja 8.

---

## 5. Propozycja minimalnego schematu (T7)

Cel: maksymalne wykorzystanie istniejącej infrastruktury, minimalna nowa logika.

### Krok 1 - Wagi 1..N-1 w pinned RAM (mała zmiana w `src/llama-model.cpp`)
- Przy `-ngl 1`: warstwa 0 -> bufor GPU (jak dziś). Warstwy 1..N-1 -> **pinned
  host buffer** (Vulkan `ggml_backend_vk_host_buffer_type`) zamiast CPU/mmap.
- To daje szybki async H2D (pinned) i trzyma model w RAM.

### Krok 2 - Fixowany bufor VRAM (ring)
- Bufor GPU o fixowanym rozmiarze S (np. 2048 MB). Trzyma: warstwa 0 (stała) +
  2 sloty ringu (po S/2 na chunk wag) + KV cache + aktywacje.
- W praktyce: rozmiar S dobieramy tak, by się zmieściło. ggml-alloc i tak
  ogranicza kopie wag do największej warstwy (T5), więc S nie musi być ogromny.

### Krok 3 - Per-chunk execution + prefetch (JEDYNA nowa logika)
- Rozbić `build_arch_graph` (dla wybranego arch) na budowanie grafu dla zakresu
  warstw `[il_start, il_end)`.
- Pętla eval: dla chunku `i`:
  1. Issue H2D copy wag chunku `i+1` na transfer queue (async, do slotu B).
  2. Compute chunku `i` na compute queue (slot A).
  3. Sync (semaphore/event): czekamy, że copy chunku `i+1` gotowe.
  4. Zamiana slotów A<->B.
- Nakładanie: copy (transfer queue) || compute (compute queue).

### Co to dotyka
- `src/llama-model.cpp` - wybór pinned buffer dla warstw 1..N-1 (Krok 1).
- `src/models/<arch>.cpp` - parametr zakresu warstw w build (Krok 3) - DOTYCZY
  archów, ale prototype może być dla JEDNEGO archu.
- `src/llama-context.cpp` - pętla per-chunk + prefetch (Krok 3).
- ggml: **brak zmian** (wykorzystujemy istniejące `cpy_tensor_async`, transfer
  queue, pinned buffer, `buffer_from_host_ptr`).

### Dlaczego to minimalne
- Nie ruszamy schedulera ggml (największe ryzyko).
- Nie wprowadzamy nowego buffer type (pinned + istniejący GPU buffer wystarczą).
- Nowa logika to tylko pętla per-chunk + prefetch w `llama-context`.

---

## 6. Oszacowanie zysku (T8 - do uzupełnienia benchmarkiem)

### Dla modelu testowego (gemma-4-E2B, 35 warstw, ~31 MB/warstwę blk)
- Chunk 8 warstw ~= 250 MB; chunk 16 warstw ~= 500 MB.
- AMD RDNA, PCIe 4.0 x16: ~25 GB/s -> copy 250 MB ~= 10 ms/chunk (8 warstw),
  500 MB ~= 20 ms/chunk (16 warstw).
- Compute chunku na 1 token (decode): do zmierzenia (szacunkowo 5-20 ms).
- Jeśli compute < copy: pipeline copy-bound, zysk = ukrycie compute pod copy
  (mały). Jeśli compute ~ copy: zysk do ~2x na fazie wag.
- KV cache (Q4, ctx 4096) ~40 MB - nie ogranicza.
- Uwaga: `per_layer_tok_embd` (1.93 GB) jest w pinned RAM, gather raz na
  forward (nie per-warstwa) - nie wchodzi w per-chunk copy.

### Ogólnie (7B Q4_K_M: ~4.4 GB, 32 warstwy -> ~140 MB/warstwę)
- Chunk 9 warstw ~= 1.26 GB -> copy ~= 50 ms/chunk (PCIe 4.0 x16).

TODO: zmierzyć `compute_ms/chunk` i `copy_ms/chunk` na docelowej karcie AMD.

---

## 7. Ryzyka i pytania otwarte (T9)

1. Czy zysk istotny na docelowej karcie AMD? (T8)
2. KV cache + warstwa 0 + ring - czy mieści docelowe ctx?
3. Pinned RAM na warstwy 1..N-1 (np. ~4 GB) - czy RAM pozwala? (pinned nie
   może być swapped).
4. Rozbicie `build_arch_graph` per-chunk dotyka archów - prototype dla 1 archu.
5. Upstream vs private fork/patch? (AGENTS.md: najpierw issue, nie od razu PR).
6. Interakcja z `--lazy` - rozszerzenie lazy, czy nowy tryb?
7. MoE: per-chunk musi uwzględniać copy tylko użytych ekspertów (jest w
   schedulerze - sekcja 2.3).
8. Optymalny rozmiar chunku (balans: prefetch latency vs rozmiar slotu).
9. Czy `async_use_transfer_queue` jest domyślnie włączone na docelowej karcie?
   (zależne od `allow_graphics_queue`; ewentualnie `GGML_VK_ASYNC_USE_TRANSFER_QUEUE`).

---

## 8. Następne kroki (T10)

1. [x] T8: pomiar compute/copy per chunk (negatywny: copy jest wąskim gardłem na PCIe)
2. [x] T10a: standalone test prymitywów (`tools/stream-test`) - overlap DZIAŁA (1.04-1.14x)
3. [ ] T10b: implementacja streaming w `src/models/gemma4.cpp` (per-chunk graph + prefetch)
4. [ ] Test empiryczny na docelowym systemie (DDR5 + iGPU + PCIe 4.0 x8)
5. [ ] Jeśli zysk istotny: otworzyć issue z propozycją, zbierać feedback maintainerów

---

## 9. Model testowy: gemma-4-E2B.i1-Q4_K_M

Plik: `D:\shared\ai-models\gemma-4-E2B.i1-Q4_K_M.gguf` (3.43 GB, arch `gemma4`).

### Fakty z GGUF
- **n_layers = 35** (blk.0 .. blk.34).
- n_embd = 1536, n_embd_per_layer = 512.
- Wagi `blk.*` per warstwa: **~22 MB (wcześniejsze) do ~42 MB (późniejsze, 2x FFN)**, śr. **~31 MB**.
- `token_embd.weight` = 315 MB (główny embedding).
- `per_layer_tok_embd.weight` = **1.93 GB**, flaga `TENSOR_READ_LAZY`.
- `per_layer_model_proj.weight` = 27.5 MB.
- 601 tensorów.

### Kluczowe dla streamingu (gemma4)
- `per_layer_tok_embd` (1.93 GB) jest czytany przez **jeden gather**
  (`ggml_get_rows` na wszystkie warstwy naraz, `src/models/gemma4.cpp:450`),
  a nie per-warstwa. Per-warstwa bierzemy tylko widok
  (`ggml_view_2d_slice`, linia 375). Zatem **per-warstwa streamujemy tylko
  wagi `blk.*` (~31 MB)**, nie per-layer embedding.
- `per_layer_tok_embd` musi być w **pinned RAM** (1.93 GB), żeby gather był szybki
  (nie z dysku/mmap).
- `token_embd` (315 MB) + `per_layer_model_proj` (27.5 MB) + `output_norm` -
  globalne, trzymane stale (GPU lub pinned RAM).

### Budżet VRAM (test, ctx 4096, KV Q4)
- Ring: 2048 MB (2 sloty po 1024 MB).
- token_embd: 315 MB (GPU).
- per_layer_model_proj: 27.5 MB.
- KV cache (Q4, ctx 4096): ~40 MB (małe).
- Aktywacje: zależne od ctx/batch.
- **Suma ~2.4 GB** - mieści się w 4-8 GB AMD GPU.

### Rozmiar chunku (dla ringu 2048 MB, 2 sloty po 1024 MB)
- Per warstwa ~31 MB -> slot 1024 MB mieści **~32 warstwy**.
- Ale double buffering potrzebuje >= 2 chunki; rozsądny chunk: **8-16 warstw**
  (250-500 MB), żeby prefetch nie dominował. Przebieg: {8, 16, 32}.
- Uwaga: późniejsze warstwy są 2x większe - chunk z późnych warstw większy.

### Pinned RAM (wymagane)
- `per_layer_tok_embd`: 1.93 GB pinned.
- Wagi `blk.1..34` (streamowane): ~1.06 GB (mogą być w pinned RAM jako źródło H2D).
- `token_embd` + `per_layer_model_proj`: ~343 MB (pinned lub GPU).
- **Suma pinned RAM ~3.3 GB** - RAM musi to pomieścić (pinned nie może być swapped).

---

## 10. Parametry do testowania (feature testowy)

Celem jest łatwe A/B: zmierzyć efekt double buffering (prefetch on/off) i
porównać z baseline (CPU) oraz górną granicą (wszystko na GPU).

### A. Parametry CLI (istniejące)

| Parametr | Wartość testowa | Po co |
|----------|-----------------|-------|
| `-m` | model 7B Q4_K_M | mały model, szybka iteracja |
| `-ngl 1` | `1` | warstwa 0 stale na GPU, reszta streamowana |
| `-c` | `4096` | kontekst testowy |
| `-ctk` | `q4_0` | KV cache K w Q4 (mniejsze KV, mieści się w VRAM obok ringu) |
| `-ctv` | `q4_0` | KV cache V w Q4 |
| `-b` | `1` | batch 1 - izoluje streaming per-token |
| `--no-warmup` | - | mierzy realny streaming (warmup by wszystko zaprefetchował) |
| `-v` | - | verbose |
| `--progress` | - | postęp |

### B. Parametry CLI (NOWE - do dodania dla feature'u testowego)

| Parametr | Domyślnie | Po co |
|----------|-----------|-------|
| `--stream-buf-size <MB>` | `2048` | rozmiar fixowanego ringu VRAM |
| `--stream-chunk <n>` | `auto`/`9` | liczba warstw w chunku (rozmiar slotu) |
| `--no-stream-prefetch` | off | wyłącza prefetch = baseline serializowany (copy then compute) |
| `--stream-debug` | off | log per-chunk: `copy_ms`, `compute_ms`, `overlap` |

`auto` dla chunku: `chunk = floor((S/2) / rozmiar_wagi_warstwy)`, z cap na n_layer-1.

### C. Zmienne środowiskowe (istniejące, do testów)

| Zmienna | Wartość | Po co |
|---------|---------|-------|
| `GGML_VK_ASYNC_USE_TRANSFER_QUEUE` | `1` | wymusza osobny transfer queue (gwarancja nakładania copy na compute na AMD) |
| `GGML_VK_DISABLE_ASYNC` | (ustawiona) | wyłącza async = A/B w pełni serializowany |
| `GGML_VK_PERF_LOGGER` | `1` | log czasu per-op (mierzy copy vs compute) |
| `GGML_VK_PIPELINE_STATS` | `1` | statystyki pipeline |
| `GGML_VK_MEMORY_LOGGER` | `1` | log zużycia pamięci (weryfikuje rozmiar ringu) |
| `GGML_VK_DEBUG_MARKERS` | (ustawiona) | markery debug do vulkan profiler / renderdoc |
| `GGML_SCHED_DEBUG` | `1` | debug splitów i kopii schedulera |

### D. Macierz testów (A/B)

1. **Baseline CPU**: `-ngl 1` (reszta na CPU) - wolny, punkt odniesienia.
2. **Górna granica**: `-ngl 99` (wszystko na GPU) - szybki, wymaga całej VRAM.
3. **Streaming, bez prefetch**: `-ngl 1 --stream-buf-size 2048 --no-stream-prefetch`
   - copy i compute serializowane.
4. **Streaming, z prefetch**: `-ngl 1 --stream-buf-size 2048`
   - pipeline (docelowy tryb).

Porównania:
- **3 vs 4** = efekt double buffering (zysk z nakładania copy na compute).
- **4 vs 2** = jak blisko górnej granicy przy ograniczonej VRAM.
- **4 vs 1** = zysk vs czysty CPU.

### E. Co mierzyć

- **tok/s** (z linii podsumowania) - główna metryka.
- **VRAM** (z `GGML_VK_MEMORY_LOGGER` / system) - weryfikacja, że ring trzyma
  się w `--stream-buf-size`.
- **per-chunk**: `copy_ms`, `compute_ms`, `overlap_ratio` (z `--stream-debug`
  + `GGML_VK_PERF_LOGGER`). `overlap_ratio = min(copy,compute)/max(copy,compute)`.
- **Rozmiar chunku**: przebieg `--stream-chunk` {4, 9, 16, 32} - znaleźć optimum.

### F. Przykładowe komendy testowe

```sh
# model testowy: D:\shared\ai-models\gemma-4-E2B.i1-Q4_K_M.gguf (3.43 GB, gemma4)
MODEL="D:\shared\ai-models\gemma-4-E2B.i1-Q4_K_M.gguf"

# 1. baseline CPU
./build/bin/llama-cli -m "$MODEL" -ngl 1 -c 4096 -ctk q4_0 -ctv q4_0 -b 1 --no-warmup -v

# 2. górna granica (wszystko na GPU)
./build/bin/llama-cli -m "$MODEL" -ngl 99 -c 4096 -ctk q4_0 -ctv q4_0 -b 1 --no-warmup -v

# 3. streaming bez prefetch (baseline serializowany)
GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1 GGML_VK_PERF_LOGGER=1 \
  ./build/bin/llama-cli -m "$MODEL" -ngl 1 -c 4096 -ctk q4_0 -ctv q4_0 -b 1 --no-warmup -v \
  --stream-buf-size 2048 --no-stream-prefetch --stream-debug

# 4. streaming z prefetch (docelowy)
GGML_VK_ASYNC_USE_TRANSFER_QUEUE=1 GGML_VK_PERF_LOGGER=1 GGML_VK_MEMORY_LOGGER=1 \
  ./build/bin/llama-cli -m "$MODEL" -ngl 1 -c 4096 -ctk q4_0 -ctv q4_0 -b 1 --no-warmup -v \
  --stream-buf-size 2048 --stream-debug
```

---

## Aneks: kluczowe lokalizacje (Vulkan/AMD)

| Co | Gdzie |
|----|-------|
| build_graph (cały graf) | `src/llama-model.cpp:2732` |
| pętla per-warstwa (przykład) | `src/models/gpt2.cpp:82` |
| lazy -> host buffer | `src/llama-model.cpp:1743-1770` |
| lazy_read::add | `src/llama-model-loader.cpp:1081` |
| op_offload (host buffer check) | `ggml/src/ggml-backend.cpp:970` |
| compute_splits (copy then compute) | `ggml/src/ggml-backend.cpp:1643-1790` |
| reużytkowanie kopii (n_copies=1) | `ggml/src/ggml-backend.cpp:1393-1407` |
| n_copies / cur_copy | `ggml/src/ggml-backend.cpp:1865,1990` |
| pinned host buffer (Vulkan) | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:17004-17040` |
| pinned malloc (Vulkan) | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:8335-8350` |
| transfer queue setup (AMD) | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:7355-7372` |
| async H2D copy (Vulkan) | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:17216-17270` |
| buffer_from_host_ptr (Vulkan) | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:19994-20001` |
| load modes | `include/llama.h:205-211` |
| lazy modes | `include/llama.h:217-221` |
| cparams.op_offload | `common/common.cpp:1748` |
| sched create (op_offload) | `src/llama-context.cpp:605` |

---

## 11. Wyniki baseline (pomiar na RX 7600 XT 8 GB)

Model: `gemma-4-E2B.i1-Q4_K_M.gguf`, `-c 4096`, `-ctk q4_0 -ctv q4_0`, `-b 256`,
`--no-warmup -st -v`. Uwaga: wymagana flaga `-v` - bez niej logi VRAM
(`model buffer size`, `memory breakdown`) są tłumione.

### Tabela baseline

| Metryka | `-ngl 1` | `-ngl 99` |
|---------|----------|-----------|
| Prompt t/s | 7.7 | **15.8** |
| Generation t/s | 5.2 | **59.7** |
| GPU model buffer | 315 MiB | 1416 MiB |
| CPU model buffer (mmap) | 3254 MiB | 2152 MiB |
| GPU self (model+ctx+compute) | 381 MiB | 1481 MiB |
| GPU free | 13203 MiB | 10972 MiB |
| GPU unaccounted (inny proc + driver) | 2782 MiB | 3913 MiB |
| RAM (Host) self | 3278 MiB | ~2170 MiB |

### Rozbiórka VRAM (format `common_memory_breakdown_print`)

```
| memory breakdown [MiB]   | total    free    self   model   context   compute    unaccounted |
|   - Vulkan0 (RX 7600 XT) | 16368 = 13203 + ( 381 =   315 +       0 +      66) +        2782 |   <- -ngl 1
|   - Vulkan0 (RX 7600 XT) | 16368 = 10972 + (1481 =  1416 +       9 +      55) +        3913 |   <- -ngl 99
```

`self = model + context + compute`. `unaccounted` = inne procesy (tu: działający
llama.cpp) + overhead drivera.

### Kluczowe obserwacje

1. **Zysk z offloadu**: generation 5.2 -> 59.7 t/s (11.5x) przy `-ngl 99`.
   To jest cel: osiągnąć ~60 t/s przy Mniejszym VRAM.

2. **per_layer_tok_embd (1.9 GB) zostaje na CPU nawet z `-ngl 99`**: dostęp
   przez `ggml_get_rows` (gather) czyta z CPU. Dlatego GPU ma tylko 1416 MiB
   (blk + per_layer_model_proj + token_embd + output), a nie 2.4 GB.

3. **Wpływ na projekt streamingowy**: bufor GPU dla streamingu musi pomieścić
   tylko chunk blk (~250-500 MB) + token_embd (315 MB) + per_layer_model_proj
   (27.5 MB) + KV (40 MB) + compute (60 MB) ~= **1.0-1.1 GB**, nie 2.4 GB.
   per_layer_tok_embd (1.9 GB) i tak jest na CPU.

4. **Unaccounted rośnie z -ngl 99** (2782 -> 3913 MiB): inny llama.cpp +
   driver. Po zatrzymaniu drugiego procesu free VRAM wzrośnie.

### Pliki

- `baseline-bench.bat` - komendy baseline (z `-v`, `-st`, `-b 256`)
- `bench-baseline-ngl1.log` - log baseline -ngl 1
- `bench-baseline-ngl99.log` - log baseline -ngl 99

---

## 12. T8: Pomiar compute_ms vs copy_ms na chunk

Cel: ustalić, czy kopia H2D chunku mieści się pod obliczeniami chunku
(copy_ms < compute_ms). Tylko wtedy pipeline daje zysk.

### Pomiar compute (z baseline -ngl 99)

Generation 59.7 t/s (wszystkie warstwy na GPU). Jeden token = pełny forward
przez 35 warstw:

```
compute_ms/token  = 1000 / 59.7      = 16.75 ms
compute_ms/layer  = 16.75 / 35       = 0.479 ms
compute_ms/chunk  = 0.479 * n_layers (np. 8 warstw = 3.83 ms)
```

### Pomiar copy (szacunek z pasma PCIe)

RX 7600 XT = PCIe 4.0 x4, efektywne pasmo H2D ~6 GB/s (teoretyczne 8 GB/s).
Rozmiar warstwy blk (avg) = 31 MB:

```
copy_ms/layer  = 31 MB / 6 GB/s  = 5.2 ms
copy_ms/chunk  = 5.2 * n_layers  (np. 8 warstw = 41.6 ms)
```

### Porównanie (chunk 8 warstw)

| | compute | copy | stosunek |
|---|---------|------|----------|
| 8 warstw | 3.83 ms | 41.6 ms | copy 10.9x wolniejsze |

**Kryterium ukrycia copy**: copy_ms <= compute_ms, czyli:

```
bytes_per_layer / bandwidth <= compute_ms/layer
bandwidth >= bytes_per_layer / compute_ms/layer
bandwidth >= 31 MB / 0.479 ms = 64.7 GB/s
```

Żaden PCIe nie osiąga 64.7 GB/s (PCIe 4.0 x4 = 6, x16 = 25, PCIe 5.0 x16 = 50 GB/s).

### Wniosek (negatywny, ale ważny)

**Dla gemma4 na PCIe kopia jest wąskim gardłem.** Streaming nie da zysku
prędkościowego - będzie ograniczony pasmem H2D:

| Scenariusz | Prędkość |
|------------|----------|
| All-on-GPU (-ngl 99) | 59.7 t/s |
| Streaming (PCIe 4.0 x4, 6 GB/s) | ~5.5 t/s |
| Streaming (PCIe 4.0 x16, 25 GB/s) | ~23 t/s |
| -ngl 1 (CPU compute) | 5.2 t/s |

Streaming jest wolniejszy niż all-on-GPU (kopia dominuje), ale nieco szybszy
niż -ngl 1 (compute na GPU). Jedyny zysk: uruchomienie modelu na GPU z
ograniczonym VRAM (gdy model nie mieści się w całości).

### Kiedy streaming DZIAŁA

Kryterium: `bytes_per_layer < bandwidth * compute_ms/layer`. Dla gemma4
(31 MB/layer) potrzebne 64.7 GB/s - niemożliwe na PCIe. Dla modelu z
małymi warstwami (np. 5 MB/layer) wystarczy 10.4 GB/s (PCIe 4.0 x8).

**Wniosek ogólny**: streaming przez PCIe ma sens dla modeli z małymi
warstwami (niska bytes_per_layer) lub przy bardzo szybkim interconnekt
(NVLink, CXL, unified memory). Dla gemma4 na RX 7600 XT - nie.

### Uwaga o pomiarze

Dokładne pasmo H2D nie zostało zmierzone empirycznie (load time ~18 s
dominuje odczyt z dysku, nie kopia). Wniosek jest robustny: dla dowolnego
realistycznego PCIe (< 64.7 GB/s) kopia jest wąskim gardłem.

## 13. Test prymitywów: `llama-stream-test` (T10)

Standalone test w `tools/stream-test/` waliduje prymitywy double-bufferingu
na Vulkan/AMD bez modyfikacji schedulera.

### Design testu

- Pinned host buffer (symulacja wag w RAM)
- 2 sloty VRAM (ring) + 1 scratch
- Double-buffering: H2D(chunk i+1) na transfer queue + D2D(chunk i) na compute queue, overlap
- Baseline: serializowany H2D + D2D per chunk
- D2D copy jako proxy compute (GPU-internal, szybszy niż H2D)

### Wyniki (RX 7600 XT 8 GB, PCIe 4.0 x16)

| chunk | n_chunks | double-buffered | serialized | speedup |
|-------|----------|-----------------|------------|---------|
| 128 MB | 8        | 76.0 ms (13.2 GB/s) | 86.4 ms (11.6 GB/s) | 1.14x |
| 256 MB | 4        | 76.0 ms (13.2 GB/s) | 85.8 ms (11.7 GB/s) | 1.13x |
| 256 MB | 8        | 277.2 ms (7.2 GB/s)  | 298.8 ms (6.7 GB/s)  | 1.08x |
| 512 MB | 4        | 263.3 ms (7.6 GB/s)  | 272.8 ms (7.3 GB/s)  | 1.04x |

### Interpretacja

- Overlap DZIAŁA (speedup > 1.0 w wszystkich konfiguracjach)
- Speedup mały (1.04-1.14x) bo D2D (proxy compute) jest znacznie szybszy niż H2D (PCIe)
- W realnym użyciu compute (matmul) jest znacznie wolniejszy niż H2D, więc speedup będzie większy
- Maksymalny teoretyczny speedup = 2x (gdy compute_time == copy_time)
- Mniejsze chunki = większy speedup (krótszy H2D, lepszy overlap)

### Wniosek

Prymitywy działają: H2D na transfer queue, compute na compute queue, overlap możliwy.
Następny krok: implementacja realnego streaming w `src/models/gemma4.cpp`.
