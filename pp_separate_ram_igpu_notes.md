# Zero-Copy i alokacja sterty Vulkan / RAM - iGPU notes

## Zero-Copy i alokacja sterty Vulkan

Aby iGPU miało natywny dostęp do KV Cache leżącego w RAM bez strat wydajności,
bufor nie może być tworzony zwykłym `malloc()`. W swoim kodzie musisz alokować go
przez API Vulkana z flagą `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` lub użyć rozszerzenia
`VK_EXT_external_memory_host` do zmapowania pamięci wskaźnikiem ze strony CPU.

## Pinowanie stron pamięci (mlock / VirtualLock)

Pamięć systemowa jest dynamicznie stronicowana przez Windows. Zanim włączysz transfery
DMA (dla RTX-a) lub natywny odczyt iGPU, bufor KV Cache musi zostać zablokowany w
fizycznych adresach RAM (`VirtualLock` na Windowsie, `mlock` na Linuxie). W przeciwnym
razie błędy stron (page faults) zamrożą wątki wykonawcze.

## Konflikty spójności cache (Cache Coherency Contention)

APU AMD posiada wspólny kontroler pamięci i spójny L3 cache. Jeśli procesor (np. sampler
lub logika sampla) modyfikuje ten sam obszar RAM, z którego Radeon 760M próbuje czytać
macierze uwagi, protokół spójności unieważnia linie pamięci podręcznej (cacheline
invalidation). Skutkuje to drastycznym spływem przepustowości z 65 GB/s do kilkunastu
GB/s.

## Wyrównanie danych (Memory Alignment)

Wszystkie bufory przeznaczone dla iGPU i silnika DMA musisz wyrównywać co najmniej do
64 bajtów (rozmiar linii cache CPU) oraz do 4 KB (rozmiar strony OS). Nieustawione
wskaźniki wymuszają na kontrolerze pamięci DDR5 wykonywanie dodatkowych cykli odczytu
(unaligned memory access penalty).
