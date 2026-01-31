// gpu_be_pcie_memcpy_task.cu
//
// Purpose: generate sustained PCIe traffic (GPU <-> CPU) to study contention.
// Patterns supported:
//   --dir=h2d   : host -> device
//   --dir=d2h   : device -> host
//   --dir=bidir : host->device + device->host (per stream)
//
// Build (SM80): nvcc gpu_be_pcie_memcpy_task.cu -O3 -std=c++14 -o gpu_pcie_memcpy \
//   -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
//
// Run example:
//   ./bin/gpu_pcie_memcpy --dir=bidir --chunk_mb=128 --streams=8 --batch=8 --pinned=1 --report_ms=1000

#include <cuda_runtime.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(cmd)                                                                           \
    do                                                                                            \
    {                                                                                             \
        cudaError_t e = (cmd);                                                                    \
        if (e != cudaSuccess)                                                                     \
        {                                                                                         \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int)
{
    g_stop = 1;
}

enum Dir
{
    DIR_H2D,
    DIR_D2H,
    DIR_BIDIR
};

struct Args
{
    int device = 0;
    double seconds = 60.0;
    size_t bytes = 0;      // if 0, use chunk_mb
    double chunk_mb = 128;  // default if bytes==0
    int streams = 8;
    int batch = 8;         // number of memcpy ops per stream per iteration
    int pinned = 1;
    int report_ms = 1000;
    int duty_on_ms = 0;
    int duty_off_ms = 0;
    Dir dir = DIR_BIDIR;
};

static size_t parse_size(const char *s)
{
    if (!s || !*s)
        return 0;
    char *end = nullptr;
    double v = strtod(s, &end);
    if (end && *end)
    {
        if (*end == 'K' || *end == 'k')
            v *= 1024.0;
        else if (*end == 'M' || *end == 'm')
            v *= 1024.0 * 1024.0;
        else if (*end == 'G' || *end == 'g')
            v *= 1024.0 * 1024.0 * 1024.0;
    }
    if (v < 0)
        v = 0;
    return (size_t)v;
}

static Dir parse_dir(const char *s)
{
    if (!s)
        return DIR_H2D;
    if (!strcmp(s, "h2d"))
        return DIR_H2D;
    if (!strcmp(s, "d2h"))
        return DIR_D2H;
    if (!strcmp(s, "bidir"))
        return DIR_BIDIR;
    return DIR_H2D;
}

static void parse_args(int argc, char **argv, Args &a)
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "--device=", 9))
            a.device = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--seconds=", 10))
            a.seconds = atof(argv[i] + 10);
        else if (!strncmp(argv[i], "--bytes=", 8))
            a.bytes = parse_size(argv[i] + 8);
        else if (!strncmp(argv[i], "--chunk_mb=", 11))
            a.chunk_mb = atof(argv[i] + 11);
        else if (!strncmp(argv[i], "--streams=", 10))
            a.streams = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--batch=", 8))
            a.batch = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--pinned=", 9))
            a.pinned = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--report_ms=", 12))
            a.report_ms = atoi(argv[i] + 12);
        else if (!strncmp(argv[i], "--duty_on_ms=", 13))
            a.duty_on_ms = atoi(argv[i] + 13);
        else if (!strncmp(argv[i], "--duty_off_ms=", 14))
            a.duty_off_ms = atoi(argv[i] + 14);
        else if (!strncmp(argv[i], "--dir=", 6))
            a.dir = parse_dir(argv[i] + 6);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
        {
            printf("Usage: %s [--device=0] [--seconds=60] [--dir=h2d|d2h|bidir]\n"
                   "           [--bytes=SIZE] [--chunk_mb=64] [--streams=4] [--batch=1]\n"
                   "           [--pinned=1] [--report_ms=1000]\n"
                   "           [--duty_on_ms=0] [--duty_off_ms=0]\n",
                   argv[0]);
            exit(0);
        }
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, on_sigint);
    Args args;
    parse_args(argc, argv, args);

    if (args.streams <= 0)
        args.streams = 1;
    if (args.batch <= 0)
        args.batch = 1;
    if (args.report_ms <= 0)
        args.report_ms = 1000;
    if (args.bytes == 0)
    {
        if (args.chunk_mb <= 0)
            args.chunk_mb = 64;
        args.bytes = (size_t)(args.chunk_mb * 1024.0 * 1024.0);
    }
    if (args.bytes == 0)
        args.bytes = 64 * 1024 * 1024;

    CUDA_CHECK(cudaSetDevice(args.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));

    const char *dir_name = (args.dir == DIR_H2D ? "h2d" : args.dir == DIR_D2H ? "d2h" : "bidir");
    printf("[pcie] device=%d name=%s dir=%s bytes=%zu streams=%d batch=%d pinned=%d report_ms=%d duty_on_ms=%d duty_off_ms=%d seconds=%.1f\n",
           args.device, prop.name, dir_name, args.bytes, args.streams, args.batch, args.pinned, args.report_ms,
           args.duty_on_ms, args.duty_off_ms, args.seconds);
    fflush(stdout);

    std::vector<void *> hbufs(args.streams, nullptr);
    std::vector<void *> dbufs(args.streams, nullptr);
    std::vector<cudaStream_t> streams(args.streams);

    for (int i = 0; i < args.streams; ++i)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        if (args.pinned)
            CUDA_CHECK(cudaHostAlloc(&hbufs[i], args.bytes, cudaHostAllocDefault));
        else
        {
            hbufs[i] = malloc(args.bytes);
            if (!hbufs[i])
            {
                fprintf(stderr, "malloc failed for host buffer\n");
                exit(1);
            }
        }
        memset(hbufs[i], 0xAB, args.bytes);
        CUDA_CHECK(cudaMalloc(&dbufs[i], args.bytes));
        CUDA_CHECK(cudaMemset(dbufs[i], 0, args.bytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup (one round)
    for (int i = 0; i < args.streams; ++i)
    {
        if (args.dir == DIR_H2D || args.dir == DIR_BIDIR)
            CUDA_CHECK(cudaMemcpyAsync(dbufs[i], hbufs[i], args.bytes, cudaMemcpyHostToDevice, streams[i]));
        if (args.dir == DIR_D2H || args.dir == DIR_BIDIR)
            CUDA_CHECK(cudaMemcpyAsync(hbufs[i], dbufs[i], args.bytes, cudaMemcpyDeviceToHost, streams[i]));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_start = std::chrono::steady_clock::now();
    auto last_report = t_start;
    uint64_t bytes_since = 0;
    uint64_t bytes_total = 0;
    bool has_duty = (args.duty_on_ms > 0 && args.duty_off_ms > 0);
    int duty_cycle_ms = args.duty_on_ms + args.duty_off_ms;

    auto sync_all = [&]() {
        for (int i = 0; i < args.streams; ++i)
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    };

    while (!g_stop)
    {
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - t_start).count();
        if (args.seconds > 0 && elapsed_s >= args.seconds)
            break;

        bool do_transfer = true;
        if (has_duty)
        {
            int elapsed_ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();
            int phase = elapsed_ms % duty_cycle_ms;
            if (phase >= args.duty_on_ms)
            {
                int off_left = duty_cycle_ms - phase;
                if (off_left > 0)
                    std::this_thread::sleep_for(std::chrono::milliseconds(std::min(off_left, 10)));
                do_transfer = false;
            }
        }

        if (do_transfer)
        {
            for (int i = 0; i < args.streams; ++i)
            {
                for (int bi = 0; bi < args.batch; ++bi)
                {
                    if (args.dir == DIR_H2D || args.dir == DIR_BIDIR)
                        CUDA_CHECK(cudaMemcpyAsync(dbufs[i], hbufs[i], args.bytes, cudaMemcpyHostToDevice, streams[i]));
                    if (args.dir == DIR_D2H || args.dir == DIR_BIDIR)
                        CUDA_CHECK(cudaMemcpyAsync(hbufs[i], dbufs[i], args.bytes, cudaMemcpyDeviceToHost, streams[i]));
                }
            }
            sync_all();

            uint64_t mult = (args.dir == DIR_BIDIR) ? 2ULL : 1ULL;
            uint64_t bytes_this = (uint64_t)args.bytes * (uint64_t)args.streams * mult * (uint64_t)args.batch;
            bytes_since += bytes_this;
            bytes_total += bytes_this;
        }

        now = std::chrono::steady_clock::now();
        auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report).count();
        if (dt_ms >= args.report_ms)
        {
            double dt_s = dt_ms / 1000.0;
            double bw_gib_s = (bytes_since / (1024.0 * 1024.0 * 1024.0)) / (dt_s > 0 ? dt_s : 1e-9);
            double t_s = std::chrono::duration<double>(now - t_start).count();
            printf("[pcie] t=%.2fs dir=%s bw_gib_s=%.3f bytes=%lu streams=%d batch=%d chunk_bytes=%zu pinned=%d\n",
                   t_s, dir_name, bw_gib_s, (unsigned long)bytes_since, args.streams, args.batch, args.bytes, args.pinned);
            fflush(stdout);
            bytes_since = 0;
            last_report = now;
        }
    }

    // Final report
    auto t_end = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();
    if (bytes_since > 0)
    {
        double dt_s = std::chrono::duration<double>(t_end - last_report).count();
        double bw_gib_s = (bytes_since / (1024.0 * 1024.0 * 1024.0)) / (dt_s > 0 ? dt_s : 1e-9);
        printf("[pcie] t=%.2fs dir=%s bw_gib_s=%.3f bytes=%lu streams=%d batch=%d chunk_bytes=%zu pinned=%d\n",
               wall_s, dir_name, bw_gib_s, (unsigned long)bytes_since, args.streams, args.batch, args.bytes, args.pinned);
        fflush(stdout);
    }
    double avg_bw = (bytes_total / (1024.0 * 1024.0 * 1024.0)) / (wall_s > 0 ? wall_s : 1e-9);
    printf("[pcie] done: wall=%.2fs total_bytes=%lu avg_bw_gib_s=%.3f\n",
           wall_s, (unsigned long)bytes_total, avg_bw);

    for (int i = 0; i < args.streams; ++i)
    {
        if (args.pinned)
            cudaFreeHost(hbufs[i]);
        else
            free(hbufs[i]);
        cudaFree(dbufs[i]);
        cudaStreamDestroy(streams[i]);
    }
    return 0;
}
