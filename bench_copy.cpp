// microbenchmark: cross-device copy patterns between two Vulkan devices
// models the --pp-dev workload: weights on b1 (iGPU), prefill compute on b0 (dGPU)
// usage: bench_copy.exe [size_mb]
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-impl.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now_ms() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char ** argv) {
    size_t SZ = 512 * 1024 * 1024;
    if (argc > 1) {
        SZ = (size_t)atol(argv[1]) * 1024 * 1024;
    }

    if (!ggml_backend_reg_count()) {
        ggml_backend_load_all();
    }
    std::vector<ggml_backend_dev_t> devs;
    for (size_t r = 0; r < ggml_backend_reg_count(); r++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(r);
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            ggml_backend_dev_t d = ggml_backend_reg_dev_get(reg, i);
            enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                devs.push_back(d);
            }
        }
    }
    if (devs.size() < 2) {
        fprintf(stderr, "need 2 GPU devices, found %zu\n", devs.size());
        return 1;
    }
    ggml_backend_dev_t d0 = devs[0];
    ggml_backend_dev_t d1 = devs[1];
    printf("dev0: %s\n", ggml_backend_dev_name(d0));
    printf("dev1: %s\n", ggml_backend_dev_name(d1));
    ggml_backend_t b0 = ggml_backend_dev_init(d0, "");
    ggml_backend_t b1 = ggml_backend_dev_init(d1, "");
    if (!b0 || !b1) {
        fprintf(stderr, "backend init failed\n");
        return 1;
    }

    // tensors: A and B are two "weight" tensors, both copied b1 -> b0 before being processed on b0
    // no_alloc: tensor data goes to backend buffers via ggml_backend_tensor_alloc, not the context pool
    struct ggml_init_params gp = { 1 << 24, nullptr, true };
    ggml_context * ctx = ggml_init(gp);

    const int64_t ncols = (int64_t)(SZ / 2) / 16384;
    auto mk = [&](ggml_context * c) {
        int64_t ne[2] = { 16384, ncols };
        return ggml_new_tensor(c, GGML_TYPE_F16, 2, ne);
    };
    ggml_tensor * sA = mk(ctx), * sB = mk(ctx);
    ggml_tensor * dA = mk(ctx), * dB = mk(ctx);

    ggml_backend_buffer_t buf_sA = ggml_backend_alloc_buffer(b1, SZ);
    ggml_backend_buffer_t buf_sB = ggml_backend_alloc_buffer(b1, SZ);
    ggml_backend_buffer_t buf_dA = ggml_backend_alloc_buffer(b0, SZ);
    ggml_backend_buffer_t buf_dB = ggml_backend_alloc_buffer(b0, SZ);
    if (!buf_sA || !buf_sB || !buf_dA || !buf_dB) {
        fprintf(stderr, "buffer alloc failed\n");
        return 1;
    }
    ggml_backend_tensor_alloc(buf_sA, sA, ggml_backend_buffer_get_base(buf_sA));
    ggml_backend_tensor_alloc(buf_sB, sB, ggml_backend_buffer_get_base(buf_sB));
    ggml_backend_tensor_alloc(buf_dA, dA, ggml_backend_buffer_get_base(buf_dA));
    ggml_backend_tensor_alloc(buf_dB, dB, ggml_backend_buffer_get_base(buf_dB));
    ggml_backend_tensor_memset(sA, 1, 0, SZ);
    ggml_backend_tensor_memset(sB, 2, 0, SZ);
    ggml_backend_synchronize(b0);
    ggml_backend_synchronize(b1);

    // compute graph: matmul with the copied tensor as weight, repeated 4x
    int64_t ne_x[2] = { 16384, 512 };
    ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F16, 2, ne_x);
    ggml_backend_buffer_t buf_x = ggml_backend_alloc_buffer(b0, ggml_nbytes(x));
    ggml_backend_tensor_alloc(buf_x, x, ggml_backend_buffer_get_base(buf_x));
    ggml_backend_tensor_memset(x, 4, 0, ggml_nbytes(x));

    // chained matmuls so the graph contains all R of them
    const int R = 4;
    ggml_tensor * yA[R], * yB[R];
    yA[0] = ggml_mul_mat(ctx, dA, x);
    yB[0] = ggml_mul_mat(ctx, dB, x);
    for (int r = 1; r < R; r++) {
        yA[r] = ggml_mul_mat(ctx, dA, yA[r - 1]);
        yB[r] = ggml_mul_mat(ctx, dB, yB[r - 1]);
    }
    ggml_cgraph * gfA = ggml_new_graph(ctx);
    ggml_cgraph * gfB = ggml_new_graph(ctx);
    ggml_build_forward_expand(gfA, yA[R - 1]);
    ggml_build_forward_expand(gfB, yB[R - 1]);
    // graph_compute does not allocate outputs; do it manually
    for (int r = 0; r < R; r++) {
        if (!yA[r]->buffer) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(b0, ggml_nbytes(yA[r]));
            ggml_backend_tensor_alloc(buf, yA[r], ggml_backend_buffer_get_base(buf));
        }
        if (!yB[r]->buffer) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(b0, ggml_nbytes(yB[r]));
            ggml_backend_tensor_alloc(buf, yB[r], ggml_backend_buffer_get_base(buf));
        }
    }

    // warmup
    ggml_backend_tensor_copy(sA, dA);
    ggml_backend_synchronize(b0);
    enum ggml_status st = ggml_backend_graph_compute(b0, gfA);
    ggml_backend_synchronize(b0);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "warmup compute failed: %s\n", ggml_status_to_string(st));
        return 1;
    }

    const int K = 4; // copies per test
    const int K2 = 2; // copy+compute rounds per test

    // T1: copy one tensor, sync, repeat
    double t = now_ms();
    for (int i = 0; i < K; i++) {
        ggml_backend_tensor_copy(sA, dA);
        ggml_backend_synchronize(b0);
    }
    double dt = now_ms() - t;
    printf("T1 sequential copy+sync:      %8.1f ms/copy  %6.2f GB/s\n", dt / K, SZ / (dt / K * 1e-3) / 1e9);

    // T2: issue two copies back to back, sync once
    t = now_ms();
    for (int i = 0; i < K / 2; i++) {
        ggml_backend_tensor_copy(sA, dA);
        ggml_backend_tensor_copy(sB, dB);
        ggml_backend_synchronize(b0);
    }
    dt = now_ms() - t;
    printf("T2 two copies in flight:      %8.1f ms/copy  %6.2f GB/s\n", dt / K, SZ / (dt / K * 1e-3) / 1e9);

    // T4c: compute only (baseline)
    t = now_ms();
    for (int i = 0; i < K2; i++) {
        ggml_backend_graph_compute(b0, gfA);
        ggml_backend_graph_compute(b0, gfB);
        ggml_backend_synchronize(b0);
    }
    dt = now_ms() - t;
    printf("T4c compute only (2 graphs):  %8.1f ms/graph\n", dt / (2 * K2));

    // T4a: copy A, compute A, copy B, compute B (alternating)
    double t_cpy = 0, t_cmp = 0;
    t = now_ms();
    for (int i = 0; i < K2; i++) {
        double p = now_ms();
        ggml_backend_tensor_copy(sA, dA);
        ggml_backend_synchronize(b0);
        t_cpy += now_ms() - p;
        p = now_ms();
        ggml_backend_graph_compute(b0, gfA);
        t_cmp += now_ms() - p;
        p = now_ms();
        ggml_backend_tensor_copy(sB, dB);
        ggml_backend_synchronize(b0);
        t_cpy += now_ms() - p;
        p = now_ms();
        ggml_backend_graph_compute(b0, gfB);
        t_cmp += now_ms() - p;
        ggml_backend_synchronize(b0);
    }
    dt = now_ms() - t;
    printf("T4a copy,compute,copy,compute %8.1f ms/round (copy %.1f ms, compute %.1f ms)\n", dt / K2, t_cpy / K2, t_cmp / K2);

    // T4b: copy A, copy B, compute A, compute B (batched copies)
    t = now_ms();
    for (int i = 0; i < K2; i++) {
        ggml_backend_tensor_copy(sA, dA);
        ggml_backend_tensor_copy(sB, dB);
        ggml_backend_synchronize(b0);
        ggml_backend_graph_compute(b0, gfA);
        ggml_backend_graph_compute(b0, gfB);
        ggml_backend_synchronize(b0);
    }
    dt = now_ms() - t;
    printf("T4b copy,copy,compute,compute %8.1f ms/round\n", dt / K2);

    // T4e: compute A, then copy B (dB untouched by compute) - is the slow copy buffer-specific?
    t = now_ms();
    ggml_backend_graph_compute(b0, gfA);
    ggml_backend_synchronize(b0);
    double p = now_ms();
    ggml_backend_tensor_copy(sB, dB);
    ggml_backend_synchronize(b0);
    printf("T4e copy after compute, clean dst: %8.1f ms\n", now_ms() - p);

    // T4d: pipelined - issue next copy while current compute runs (no sync between them)
    t = now_ms();
    for (int i = 0; i < K2; i++) {
        ggml_backend_graph_compute_async(b0, gfA);
        ggml_backend_tensor_copy(sB, dB);   // issued while compute A is in flight
        ggml_backend_synchronize(b0);
        ggml_backend_graph_compute_async(b0, gfB);
        ggml_backend_tensor_copy(sA, dA);
        ggml_backend_synchronize(b0);
    }
    dt = now_ms() - t;
    printf("T4d pipelined copy||compute:  %8.1f ms/round\n", dt / K2);

    printf("tensor size: %zu MiB, matmul: %dx%dx512 x%d, R=%d\n", SZ / (1024 * 1024), 16384, (int)ncols, (int)ncols, R);
    return 0;
}
