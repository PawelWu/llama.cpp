// PoC: split prefill (PP) and decode across two devices, with a one-shot KV transfer in between.
//
// The model is loaded twice:
//   - PP model:     n_gpu_layers = --pp-ngl, placed on the PP device
//   - decode model: n_gpu_layers = -1 (all), placed on the decode device
//
// Flow:
//   1. prefill the prompt on the PP context
//   2. sample the first token from the PP context
//   3. copy the KV state (seq 0) from the PP context to the decode context via a host buffer
//   4. run the decode loop on the decode context
//
// The KV transfer uses llama_state_seq_get_data_ext / llama_state_seq_set_data_ext, which is the
// checkpointing API: a save+load round-trip preserves the exact sequence state (cells + positions),
// so the decode context continues from where the prompt left off.
//
// The prompt is run through the model's chat template (if it has one) unless --raw is given.
// Sampling defaults to greedy (deterministic); set --temp > 0 for top-k/min-p sampling.
//
// Usage:
//   pp-sep -m model.gguf --list-devs
//   pp-sep -m model.gguf --pp-dev <name> --dec-dev <name> [--pp-ngl K] [-n N] [prompt...]
//
// <name> is the device name shown by --list-devs (e.g. CUDA0, Vulkan0, Vulkan1, CPU).

#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>

static const char * dev_type_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
        default: return "?";
    }
}

static void list_devices() {
    const size_t n = ggml_backend_dev_count();
    printf("devices:\n");
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        printf("  [%zu] %-5s %-14s (%s) free=%zuMB total=%zuMB\n",
            i, dev_type_name(type), name, desc, free_mem / 1024 / 1024, total_mem / 1024 / 1024);
    }
}

static ggml_backend_dev_t find_device_by_name(const char * name) {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(dev), name) == 0) {
            return dev;
        }
    }
    return nullptr;
}

static double now_ms() {
    using namespace std::chrono;
    return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count() * 1000.0;
}

static llama_model * load_model_on_device(const std::string & path, ggml_backend_dev_t dev, int32_t n_gpu_layers) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu     = 0;
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices = devs;
    llama_model * m = llama_model_load_from_file(path.c_str(), mp);
    if (!m) {
        fprintf(stderr, "error: failed to load model on device\n");
        exit(1);
    }
    return m;
}

static void print_token(const struct llama_vocab * vocab, llama_token tok) {
    char buf[128];
    int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
    if (n > 0) {
        fwrite(buf, 1, n, stdout);
    }
    fflush(stdout);
}

// build the final prompt string: run the user prompt through the model's chat template
// (with an optional system message) unless raw is set or the model has no template
static std::string build_prompt(const llama_model * model, const std::string & system, const std::string & user, bool raw) {
    const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);
    if (raw || !tmpl) {
        std::string p = system;
        if (!p.empty() && !user.empty()) p += " ";
        p += user;
        return p;
    }
    std::vector<llama_chat_message> messages;
    if (!system.empty()) {
        messages.push_back({"system", strdup(system.c_str())});
    }
    messages.push_back({"user", strdup(user.c_str())});
    std::vector<char> buf(8192);
    int len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, buf.data(), (int) buf.size());
    if (len > (int) buf.size()) {
        buf.resize(len);
        len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, buf.data(), (int) buf.size());
    }
    if (len < 0) {
        fprintf(stderr, "error: failed to apply chat template\n");
        exit(1);
    }
    std::string result(buf.data(), len);
    for (auto & m : messages) {
        free(const_cast<char *>(m.content));
    }
    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_ALL, "");
    setvbuf(stdout, NULL, _IONBF, 0);

    std::string model_path;
    std::string pp_dev, dec_dev;
    std::string system_prompt = "You are a helpful assistant.";
    int32_t pp_ngl = -1;
    int32_t n_predict = 32;
    int32_t n_ctx = 0; // 0 == auto (prompt + n_predict)
    bool list_devs = false;
    bool raw = false;
    // sampling
    float    temp  = 0.0f;
    int32_t  top_k = 40;
    float    top_p = 0.9f;
    float    min_p = 0.05f;
    uint32_t seed  = 0;
    std::vector<std::string> prompt_args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--list-devs") {
            list_devs = true;
        } else if (arg == "--pp-dev" && i + 1 < argc) {
            pp_dev = argv[++i];
        } else if (arg == "--dec-dev" && i + 1 < argc) {
            dec_dev = argv[++i];
        } else if (arg == "--pp-ngl" && i + 1 < argc) {
            pp_ngl = std::stoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (arg == "--raw") {
            raw = true;
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if (arg == "--min-p" && i + 1 < argc) {
            min_p = std::stof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            printf("usage: %s -m model.gguf --list-devs\n", argv[0]);
            printf("       %s -m model.gguf --pp-dev <name> --dec-dev <name> [--pp-ngl K] [-n N] [-c N] "
                   "[--system S] [--raw] [--temp T] [--top-k K] [--top-p P] [--min-p P] [--seed S] [prompt...]\n", argv[0]);
            return 0;
        } else {
            prompt_args.push_back(arg);
        }
    }

    ggml_backend_load_all();

    if (list_devs) {
        list_devices();
        return 0;
    }

    if (model_path.empty() || pp_dev.empty() || dec_dev.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --pp-dev <name> --dec-dev <name> [--pp-ngl K] [-n N] [prompt...]\n", argv[0]);
        fprintf(stderr, "       %s -m model.gguf --list-devs\n", argv[0]);
        return 1;
    }

    ggml_backend_dev_t dev_pp  = find_device_by_name(pp_dev.c_str());
    ggml_backend_dev_t dev_dec = find_device_by_name(dec_dev.c_str());
    if (!dev_pp) {
        fprintf(stderr, "error: PP device '%s' not found\n", pp_dev.c_str());
        list_devices();
        return 1;
    }
    if (!dev_dec) {
        fprintf(stderr, "error: decode device '%s' not found\n", dec_dev.c_str());
        list_devices();
        return 1;
    }
    printf("PP device:  %s (%s)\n", ggml_backend_dev_name(dev_pp),  ggml_backend_dev_description(dev_pp));
    printf("DEC device: %s (%s)\n", ggml_backend_dev_name(dev_dec), ggml_backend_dev_description(dev_dec));

    // join the prompt args into a single user prompt
    std::string user_prompt;
    for (const auto & a : prompt_args) {
        if (!user_prompt.empty()) user_prompt += " ";
        user_prompt += a;
    }

    // load the model twice, each pinned to a single device
    printf("loading PP model (n_gpu_layers = %d)...\n", pp_ngl);
    llama_model * model_pp  = load_model_on_device(model_path, dev_pp,  pp_ngl);
    printf("loading decode model (n_gpu_layers = -1)...\n");
    llama_model * model_dec = load_model_on_device(model_path, dev_dec, -1);

    const struct llama_vocab * vocab = llama_model_get_vocab(model_pp);

    // build the prompt (chat template applied unless --raw)
    std::string prompt = build_prompt(model_pp, system_prompt, user_prompt, raw);
    printf("prompt (%zu chars):\n%s\n\n", prompt.size(), prompt.c_str());

    // tokenize
    std::vector<llama_token> prompt_tokens;
    {
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        prompt_tokens.resize(n_vocab);
        // with a chat template the special tokens are already in the string, so interpret them
        // and do not add a BOS (the template provides the framing); raw keeps the old behavior
        const bool add_bos  = raw;
        const bool special  = !raw;
        int32_t n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), prompt_tokens.data(), n_vocab, add_bos, special);
        if (n < 0) {
            fprintf(stderr, "error: tokenization failed (need %d tokens)\n", -n);
            return 1;
        }
        prompt_tokens.resize(n);
    }
    printf("prompt: %zu tokens\n", prompt_tokens.size());

    // create the two contexts
    llama_context_params cp = llama_context_default_params();
    if (n_ctx > 0) {
        cp.n_ctx = (uint32_t) n_ctx;
    } else {
        cp.n_ctx = (uint32_t) prompt_tokens.size() + (uint32_t) n_predict;
    }
    printf("creating PP context (n_ctx = %u)...\n", cp.n_ctx);
    llama_context * ctx_pp  = llama_init_from_model(model_pp,  cp);
    if (!ctx_pp)  { fprintf(stderr, "error: failed to create PP context\n");  return 1; }
    printf("creating decode context (n_ctx = %u)...\n", cp.n_ctx);
    llama_context * ctx_dec = llama_init_from_model(model_dec, cp);
    if (!ctx_dec) { fprintf(stderr, "error: failed to create decode context\n"); return 1; }

    // sampler: greedy by default (deterministic), top-k/min-p/temp when --temp > 0
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }

    // 1. prefill the prompt on the PP context
    double t0 = now_ms();
    llama_batch batch_pp = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    if (llama_decode(ctx_pp, batch_pp) != 0) {
        fprintf(stderr, "error: PP prefill failed\n");
        return 1;
    }
    double t_prefill = now_ms() - t0;
    printf("PP prefill: %zu tokens in %.1f ms (%.1f t/s)\n",
        prompt_tokens.size(), t_prefill, 1000.0 * prompt_tokens.size() / t_prefill);

    // 2. sample the first token from the PP context
    llama_token cur = llama_sampler_sample(smpl, ctx_pp, -1);
    printf("first token: %d\n", cur);

    // 3. transfer the KV state (seq 0) from the PP context to the decode context
    //    using the dedicated copy: only the first n_prompt cells of stream 0, no metadata
    t0 = now_ms();
    llama_kv_cache_copy_from(ctx_dec, ctx_pp, (uint32_t) prompt_tokens.size());
    double t_transfer = now_ms() - t0;
    printf("KV transfer: %zu cells in %.1f ms\n", prompt_tokens.size(), t_transfer);

    // 4. decode loop on the decode context
    printf("decoding up to %d tokens on the decode device:\n", n_predict);
    llama_batch batch_dec = llama_batch_get_one(&cur, 1);
    int n_dec = 0;
    t0 = now_ms();
    for (int i = 0; i < n_predict; i++) {
        if (llama_decode(ctx_dec, batch_dec) != 0) {
            fprintf(stderr, "error: decode failed at step %d\n", i);
            return 1;
        }
        llama_token next = llama_sampler_sample(smpl, ctx_dec, -1);
        if (llama_vocab_is_eog(vocab, next)) {
            break;
        }
        print_token(vocab, next);
        n_dec++;
        batch_dec = llama_batch_get_one(&next, 1);
    }
    double t_decode = now_ms() - t0;
    printf("\n");
    printf("decode: %d tokens in %.1f ms (%.1f t/s)\n", n_dec, t_decode, 1000.0 * n_dec / t_decode);

    printf("\nsummary:\n");
    printf("  prefill (PP):  %.1f ms\n", t_prefill);
    printf("  KV transfer:   %.1f ms\n", t_transfer);
    printf("  decode:        %.1f ms\n", t_decode);

    llama_sampler_free(smpl);
    llama_free(ctx_pp);
    llama_free(ctx_dec);
    llama_model_free(model_pp);
    llama_model_free(model_dec);

    return 0;
}
