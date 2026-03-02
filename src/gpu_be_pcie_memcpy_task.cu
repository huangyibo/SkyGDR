// gpu_be_pcie_memcpy_task.cu
//
// 目标：
//   持续制造 GPU<->CPU 的 PCIe 传输压力，用于 Contention 2（PCIe Fabric）实验。
//
// 支持方向：
//   --dir=h2d   : Host -> Device（CPU 内存写入 GPU）
//   --dir=d2h   : Device -> Host（GPU 读回 CPU 内存）
//
// 本实现的关键优化：
//   1) 使用 event ring + inflight 窗口来限制每个 stream 的在途批次。
//   2) 调度上采用“先 query 再提交”的非阻塞策略，而不是每轮都同步所有 stream。
//   3) 这样能避免单个慢 stream 造成全局 head-of-line blocking，减少 DMA engine 气泡。
//
// 关键参数的调优顺序（建议）：
//   1) 先固定 chunk 大小：--chunk_mb=128 或 256（过小会被提交开销吞噬，过大可能降低并行度）。
//   2) 再调 --streams：先 8，再试 12/16；观察 util 是否继续上升。
//   3) 再调 --inflight：建议 4/8/16；太小易气泡，太大可能增加排队抖动。
//   4) 再调 --batch：建议 2/4/8；batch 太大可能放大单次批次时延，不利于细粒度调度。
//
// 为什么常见只能到 90%+ 而非 100%：
//   - logger 的 utilization 是“有效负载 / 理论链路带宽”的估算值；
//   - 实际还存在协议开销、事务粒度、驱动/调度开销、计量窗口平滑；
//   - 所以 88~95% 往往已经接近该平台可达上限，Replay=0 说明链路本身健康。
//
// 编译示例（SM80）：
//   nvcc gpu_be_pcie_memcpy_task.cu -O3 -std=c++14 -o gpu_pcie_memcpy \
//     -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
//
// 运行示例：
//   ./bin/gpu_pcie_memcpy --dir=d2h --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=1000

#include <cuda_runtime.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

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
    DIR_D2H
};

struct Args
{
    int device = 0;
    double seconds = 60.0;
    size_t bytes = 0;      // 若为 0，则使用 chunk_mb 推导
    double chunk_mb = 128; // bytes=0 时的默认块大小（MiB）
    int streams = 8;
    int batch = 8;         // 每次给单个 stream 提交多少个 memcpy
    int inflight = 8;      // 每个 stream 允许的“在途批次”窗口（event ring 深度）
    int pinned = 1;
    int report_ms = 1000;
    int duty_on_ms = 0;
    int duty_off_ms = 0;
    Dir dir = DIR_D2H;
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
    return DIR_D2H;
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
        else if (!strncmp(argv[i], "--inflight=", 11))
            a.inflight = atoi(argv[i] + 11);
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
            printf("Usage: %s [--device=0] [--seconds=60] [--dir=h2d|d2h]\n"
                   "           [--bytes=SIZE] [--chunk_mb=128] [--streams=8] [--batch=8]\n"
                   "           [--inflight=8] (batches in-flight per stream; higher reduces bubbles)\n"
                   "           [--pinned=1] [--report_ms=1000]\n"
                   "           [--duty_on_ms=0] [--duty_off_ms=0]\n",
                   argv[0]);
            exit(0);
        }
    }
}

static inline void decide_dir(const Args &args, bool &do_h2d, bool &do_d2h)
{
    do_h2d = (args.dir == DIR_H2D);
    do_d2h = (args.dir == DIR_D2H);
}

static inline uint64_t stream_bytes_per_issue(const Args &args)
{
    bool do_h2d = false, do_d2h = false;
    decide_dir(args, do_h2d, do_d2h);
    uint64_t dir_mult = (do_h2d ? 1ULL : 0ULL) + (do_d2h ? 1ULL : 0ULL);
    return (uint64_t)args.bytes * (uint64_t)args.batch * dir_mult;
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
    if (args.inflight <= 0)
        args.inflight = 1;
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

    const char *dir_name = (args.dir == DIR_H2D ? "h2d" : "d2h");
    printf("[pcie] device=%d name=%s dir=%s bytes=%zu streams=%d batch=%d inflight=%d pinned=%d report_ms=%d duty_on_ms=%d duty_off_ms=%d seconds=%.1f\n",
           args.device, prop.name, dir_name, args.bytes, args.streams, args.batch, args.inflight, args.pinned, args.report_ms,
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

    // 为每个 stream 创建 event ring，用于实现 inflight 窗口限流：
    // - 每个 slot 对应“该 slot 上次提交的批次是否完成”；
    // - 新提交前先检查该 slot 是否完成，避免无限制堆积导致不可控排队。
    std::vector<std::vector<cudaEvent_t>> done(args.streams);
    std::vector<int> ev_head(args.streams, 0);

    for (int i = 0; i < args.streams; ++i)
    {
        done[i].resize(args.inflight);
        for (int k = 0; k < args.inflight; ++k)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&done[i][k], cudaEventDisableTiming));
            // 初始先打标为“已完成”，保证第一轮提交不会被无意义阻塞。
            CUDA_CHECK(cudaEventRecord(done[i][k], streams[i]));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 预热一轮：让上下文/路径稳定，减少首轮抖动对统计窗口的污染。
    for (int i = 0; i < args.streams; ++i)
    {
        bool do_h2d = false, do_d2h = false;
        decide_dir(args, do_h2d, do_d2h);

        if (do_h2d)
            CUDA_CHECK(cudaMemcpyAsync(dbufs[i], hbufs[i], args.bytes, cudaMemcpyHostToDevice, streams[i]));
        if (do_d2h)
            CUDA_CHECK(cudaMemcpyAsync(hbufs[i], dbufs[i], args.bytes, cudaMemcpyDeviceToHost, streams[i]));
        CUDA_CHECK(cudaEventRecord(done[i][0], streams[i]));
        ev_head[i] = 1 % args.inflight;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t_start = std::chrono::steady_clock::now(); // 压测起点
    auto last_report = t_start;                      // 上次打印统计的时间点
    uint64_t bytes_since = 0;                        // 当前统计窗口内已提交字节数
    uint64_t bytes_total = 0;                        // 全程累计已提交字节数

    bool has_duty = (args.duty_on_ms > 0 && args.duty_off_ms > 0);
    int duty_cycle_ms = args.duty_on_ms + args.duty_off_ms;
    int rr_wait_idx = 0; // 当所有 stream 都暂不可提交时，用 round-robin 选一个阻塞等待

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
            // 非阻塞调度策略：
            // 1) 先对每个 stream 的当前 slot 做 cudaEventQuery（不阻塞）；
            // 2) 只有 slot 完成才提交下一批；
            // 3) 若本轮一个都提交不了，再阻塞等待一个 slot 完成，避免空转占满 CPU。
            bool issued_any = false;
            for (int i = 0; i < args.streams; ++i)
            {
                bool do_h2d = false, do_d2h = false;
                decide_dir(args, do_h2d, do_d2h);

                int k = ev_head[i];
                cudaError_t q = cudaEventQuery(done[i][k]);
                if (q == cudaErrorNotReady)
                    continue;
                if (q != cudaSuccess)
                    CUDA_CHECK(q);

                for (int bi = 0; bi < args.batch; ++bi)
                {
                    if (do_h2d)
                        CUDA_CHECK(cudaMemcpyAsync(dbufs[i], hbufs[i], args.bytes, cudaMemcpyHostToDevice, streams[i]));
                    if (do_d2h)
                        CUDA_CHECK(cudaMemcpyAsync(hbufs[i], dbufs[i], args.bytes, cudaMemcpyDeviceToHost, streams[i]));
                }

                // 在该 slot 上记录“本批次结束”事件，供后续复用此 slot 前判断是否完成。
                CUDA_CHECK(cudaEventRecord(done[i][k], streams[i]));
                ev_head[i] = (k + 1) % args.inflight;

                uint64_t issued_bytes = stream_bytes_per_issue(args);
                bytes_since += issued_bytes;
                bytes_total += issued_bytes;
                issued_any = true;
            }

            if (!issued_any)
            {
                int i = rr_wait_idx % args.streams;
                int k = ev_head[i];
                CUDA_CHECK(cudaEventSynchronize(done[i][k]));
                rr_wait_idx = (rr_wait_idx + 1) % args.streams;
            }
        }

        now = std::chrono::steady_clock::now();
        auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_report).count();
        if (dt_ms >= args.report_ms)
        {
            double dt_s = dt_ms / 1000.0;
            double bw_gib_s = (bytes_since / (1024.0 * 1024.0 * 1024.0)) / (dt_s > 0 ? dt_s : 1e-9);
            double t_s = std::chrono::duration<double>(now - t_start).count();
            printf("[pcie] t=%.2fs dir=%s bw_gib_s=%.3f bytes=%lu streams=%d batch=%d inflight=%d chunk_bytes=%zu pinned=%d\n",
                   t_s, dir_name, bw_gib_s, (unsigned long)bytes_since, args.streams, args.batch, args.inflight, args.bytes, args.pinned);
            fflush(stdout);
            bytes_since = 0;
            last_report = now;
        }
    }

    // 退出前等待所有在途 memcpy 完成，保证统计与资源释放正确。
    CUDA_CHECK(cudaDeviceSynchronize());

    // 最终汇总
    auto t_end = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();
    double avg_bw = (bytes_total / (1024.0 * 1024.0 * 1024.0)) / (wall_s > 0 ? wall_s : 1e-9);
    printf("[pcie] done: wall=%.2fs total_bytes=%lu avg_bw_gib_s=%.3f\n",
           wall_s, (unsigned long)bytes_total, avg_bw);

    for (int i = 0; i < args.streams; ++i)
    {
        for (int k = 0; k < args.inflight; ++k)
            cudaEventDestroy(done[i][k]);

        if (args.pinned)
            cudaFreeHost(hbufs[i]);
        else
            free(hbufs[i]);
        cudaFree(dbufs[i]);
        cudaStreamDestroy(streams[i]);
    }
    return 0;
}
