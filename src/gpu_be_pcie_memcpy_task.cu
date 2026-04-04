// gpu_be_pcie_memcpy_task.cu
//
// 目标：
//   持续制造 GPU<->CPU 的 PCIe 传输压力，用于 Contention 2（PCIe Fabric）实验。
//
// 支持方向：
//   --dir=h2d   : Host -> Device（CPU 内存写入 GPU）
//   --dir=d2h   : Device -> Host（GPU 读回 CPU 内存）
//
// 编译：优先使用 Makefile，让 CUDA/HIP 编译器自动选择。
//
// 运行示例：
//   ./bin/gpu_pcie_memcpy --dir=d2h --chunk_mb=128 --streams=8 --batch=8 --inflight=8 --pinned=1 --report_ms=1000

#include "gpu_rt.h"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

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
    size_t total_bytes = 0; // 0 表示无限压测；>0 表示模拟有限 restore/copy window
    size_t max_outstanding_bytes = 0; // 0 表示不额外限制；>0 限制全局已提交未完成字节数
    int progress_ms = 0;    // 0 -> follow report_ms
    int progress_smooth_windows = 5; // 输出平滑带宽时使用的窗口数
    std::string progress_out;
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
        else if (!strncmp(argv[i], "--total_bytes=", 14))
            a.total_bytes = parse_size(argv[i] + 14);
        else if (!strncmp(argv[i], "--max_outstanding_bytes=", 24))
            a.max_outstanding_bytes = parse_size(argv[i] + 24);
        else if (!strncmp(argv[i], "--progress_ms=", 14))
            a.progress_ms = atoi(argv[i] + 14);
        else if (!strncmp(argv[i], "--progress_smooth_windows=", 26))
            a.progress_smooth_windows = atoi(argv[i] + 26);
        else if (!strncmp(argv[i], "--progress_out=", 15))
            a.progress_out = argv[i] + 15;
        else if (!strncmp(argv[i], "--dir=", 6))
            a.dir = parse_dir(argv[i] + 6);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
        {
            printf("Usage: %s [--device=0] [--seconds=60] [--dir=h2d|d2h]\n"
                   "           [--bytes=SIZE] [--chunk_mb=128] [--streams=8] [--batch=8]\n"
                   "           [--inflight=8] (batches in-flight per stream; higher reduces bubbles)\n"
                   "           [--pinned=1] [--report_ms=1000] [--total_bytes=0]\n"
                   "           [--max_outstanding_bytes=0]\n"
                   "           [--progress_ms=report_ms] [--progress_smooth_windows=5] [--progress_out=PATH]\n"
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
    if (args.progress_ms <= 0)
        args.progress_ms = args.report_ms;
    if (args.progress_smooth_windows <= 0)
        args.progress_smooth_windows = 1;

    if (args.bytes == 0)
    {
        if (args.chunk_mb <= 0)
            args.chunk_mb = 64;
        args.bytes = (size_t)(args.chunk_mb * 1024.0 * 1024.0);
    }
    if (args.bytes == 0)
        args.bytes = 64 * 1024 * 1024;
    if (args.total_bytes > 0 && args.max_outstanding_bytes == 0)
    {
        uint64_t default_cap = (uint64_t)args.bytes * (uint64_t)std::max(args.streams, 1) * 2ULL;
        uint64_t floor_cap = 512ULL * 1024ULL * 1024ULL;
        args.max_outstanding_bytes = (size_t)std::max(default_cap, floor_cap);
    }

    GPU_RT_CHECK(gpuSetDevice(args.device));

    gpuDeviceProp prop{};
    GPU_RT_CHECK(gpuGetDeviceProperties(&prop, args.device));

    const char *dir_name = (args.dir == DIR_H2D ? "h2d" : "d2h");
    printf("[pcie] device=%d name=%s dir=%s bytes=%zu streams=%d batch=%d inflight=%d pinned=%d report_ms=%d progress_ms=%d total_bytes=%zu max_outstanding_bytes=%zu duty_on_ms=%d duty_off_ms=%d seconds=%.1f\n",
           args.device, prop.name, dir_name, args.bytes, args.streams, args.batch, args.inflight, args.pinned, args.report_ms,
           args.progress_ms, args.total_bytes, args.max_outstanding_bytes, args.duty_on_ms, args.duty_off_ms, args.seconds);
    fflush(stdout);

    std::vector<void *> hbufs(args.streams, nullptr);
    std::vector<void *> dbufs(args.streams, nullptr);
    std::vector<gpuStream_t> streams(args.streams);

    for (int i = 0; i < args.streams; ++i)
    {
        GPU_RT_CHECK(gpuStreamCreateWithFlags(&streams[i], gpuStreamNonBlocking));
        if (args.pinned)
            GPU_RT_CHECK(gpuHostAlloc(&hbufs[i], args.bytes, gpuHostAllocDefault));
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
        GPU_RT_CHECK(gpuMalloc(&dbufs[i], args.bytes));
        GPU_RT_CHECK(gpuMemset(dbufs[i], 0, args.bytes));
    }
    GPU_RT_CHECK(gpuDeviceSynchronize());

    // 为每个 stream 创建 event ring，用于实现 inflight 窗口限流：
    // - 每个 slot 对应“该 slot 上次提交的批次是否完成”；
    // - 新提交前先检查该 slot 是否完成，避免无限制堆积导致不可控排队。
    std::vector<std::vector<gpuEvent_t>> done(args.streams);
    std::vector<int> ev_head(args.streams, 0);
    std::vector<std::vector<uint64_t>> pending_bytes(args.streams, std::vector<uint64_t>(args.inflight, 0));

    for (int i = 0; i < args.streams; ++i)
    {
        done[i].resize(args.inflight);
        for (int k = 0; k < args.inflight; ++k)
        {
            GPU_RT_CHECK(gpuEventCreateWithFlags(&done[i][k], gpuEventDisableTiming));
            // 初始先打标为“已完成”，保证第一轮提交不会被无意义阻塞。
            GPU_RT_CHECK(gpuEventRecord(done[i][k], streams[i]));
        }
    }
    GPU_RT_CHECK(gpuDeviceSynchronize());

    FILE *progress_fp = nullptr;
    if (!args.progress_out.empty())
    {
        progress_fp = fopen(args.progress_out.c_str(), "w");
        if (!progress_fp)
        {
            fprintf(stderr, "failed to open progress_out=%s\n", args.progress_out.c_str());
            exit(1);
        }
        fprintf(progress_fp, "ts_unix_ms,elapsed_ms,issued_bytes,completed_bytes,remaining_bytes,window_bytes,inst_bw_gib_s,smooth_bw_gib_s,avg_bw_gib_s,done\n");
        fflush(progress_fp);
    }

    // 预热一轮：让上下文/路径稳定，减少首轮抖动对统计窗口的污染。
    for (int i = 0; i < args.streams; ++i)
    {
        bool do_h2d = false, do_d2h = false;
        decide_dir(args, do_h2d, do_d2h);

        if (do_h2d)
            GPU_RT_CHECK(gpuMemcpyAsync(dbufs[i], hbufs[i], args.bytes, gpuMemcpyHostToDevice, streams[i]));
        if (do_d2h)
            GPU_RT_CHECK(gpuMemcpyAsync(hbufs[i], dbufs[i], args.bytes, gpuMemcpyDeviceToHost, streams[i]));
        GPU_RT_CHECK(gpuEventRecord(done[i][0], streams[i]));
        ev_head[i] = 1 % args.inflight;
    }
    GPU_RT_CHECK(gpuDeviceSynchronize());

    auto t_start = std::chrono::steady_clock::now(); // 压测起点
    auto last_progress = t_start;
    uint64_t issued_total = 0;       // 全程累计已提交字节数
    uint64_t completed_total = 0;    // 全程累计已完成字节数
    uint64_t completed_since = 0;    // 当前进度窗口内已完成字节数
    std::deque<std::pair<uint64_t, uint64_t>> smooth_hist;
    uint64_t smooth_bytes_sum = 0;
    uint64_t smooth_ms_sum = 0;
    double last_smooth_bw_gib_s = 0.0;

    bool has_duty = (args.duty_on_ms > 0 && args.duty_off_ms > 0);
    int duty_cycle_ms = args.duty_on_ms + args.duty_off_ms;
    int rr_wait_idx = 0; // 当所有 stream 都暂不可提交时，用 round-robin 选一个阻塞等待
    auto settle_slot = [&](int stream_idx, int slot_idx) {
        uint64_t finished = pending_bytes[stream_idx][slot_idx];
        if (finished == 0)
            return;
        pending_bytes[stream_idx][slot_idx] = 0;
        completed_total += finished;
        completed_since += finished;
    };
    auto poll_ready_slots = [&]() -> bool {
        bool settled_any = false;
        for (int i = 0; i < args.streams; ++i)
        {
            for (int k = 0; k < args.inflight; ++k)
            {
                if (pending_bytes[i][k] == 0)
                    continue;
                gpuError_t q = gpuEventQuery(done[i][k]);
                if (q == gpuErrorNotReady)
                    continue;
                if (q != gpuSuccess)
                    GPU_RT_CHECK(q);
                settle_slot(i, k);
                settled_any = true;
            }
        }
        return settled_any;
    };
    auto wait_one_pending_slot = [&]() -> bool {
        for (int step = 0; step < args.streams; ++step)
        {
            int i = (rr_wait_idx + step) % args.streams;
            for (int k = 0; k < args.inflight; ++k)
            {
                if (pending_bytes[i][k] == 0)
                    continue;
                GPU_RT_CHECK(gpuEventSynchronize(done[i][k]));
                settle_slot(i, k);
                rr_wait_idx = (i + 1) % args.streams;
                return true;
            }
        }
        return false;
    };
    auto flush_progress = [&](bool force_done) {
        auto now = std::chrono::steady_clock::now();
        auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count();
        if (!force_done && dt_ms < args.progress_ms)
            return;
        if (dt_ms <= 0)
            dt_ms = args.progress_ms > 0 ? args.progress_ms : 1;
        double dt_s = dt_ms / 1000.0;
        if (dt_s <= 0.0)
            dt_s = 1e-9;
        uint64_t remaining = 0;
        if (args.total_bytes > 0 && completed_total < args.total_bytes)
            remaining = args.total_bytes - completed_total;
        bool terminal_marker_only = force_done && args.progress_ms > 0 && dt_ms < args.progress_ms;
        uint64_t window_bytes = terminal_marker_only ? 0 : completed_since;
        double inst_bw_gib_s = terminal_marker_only ? 0.0 : (window_bytes / (1024.0 * 1024.0 * 1024.0)) / dt_s;
        if (!terminal_marker_only)
        {
            smooth_hist.emplace_back(window_bytes, (uint64_t)dt_ms);
            smooth_bytes_sum += window_bytes;
            smooth_ms_sum += (uint64_t)dt_ms;
            while ((int)smooth_hist.size() > args.progress_smooth_windows)
            {
                smooth_bytes_sum -= smooth_hist.front().first;
                smooth_ms_sum -= smooth_hist.front().second;
                smooth_hist.pop_front();
            }
            double smooth_dt_s = smooth_ms_sum > 0 ? (smooth_ms_sum / 1000.0) : dt_s;
            if (smooth_dt_s <= 0.0)
                smooth_dt_s = 1e-9;
            last_smooth_bw_gib_s = (smooth_bytes_sum / (1024.0 * 1024.0 * 1024.0)) / smooth_dt_s;
        }
        double smooth_bw_gib_s = terminal_marker_only ? 0.0 : last_smooth_bw_gib_s;
        double elapsed_s = std::chrono::duration<double>(now - t_start).count();
        double avg_bw_gib_s = (completed_total / (1024.0 * 1024.0 * 1024.0)) / (elapsed_s > 0 ? elapsed_s : 1e-9);
        uint64_t elapsed_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count();
        printf("[pcie] t=%.2fs dir=%s bw_gib_s=%.3f bytes=%lu streams=%d batch=%d inflight=%d chunk_bytes=%zu pinned=%d issued_total=%lu completed_total=%lu remaining_bytes=%lu\n",
               elapsed_s, dir_name, inst_bw_gib_s, (unsigned long)window_bytes, args.streams, args.batch, args.inflight,
               args.bytes, args.pinned, (unsigned long)issued_total, (unsigned long)completed_total, (unsigned long)remaining);
        fflush(stdout);
        if (progress_fp)
        {
            uint64_t ts_unix_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::system_clock::now().time_since_epoch())
                                      .count();
            fprintf(progress_fp, "%lu,%lu,%lu,%lu,%lu,%lu,%.6f,%.6f,%.6f,%d\n",
                    (unsigned long)ts_unix_ms,
                    (unsigned long)elapsed_ms,
                    (unsigned long)issued_total,
                    (unsigned long)completed_total,
                    (unsigned long)remaining,
                    (unsigned long)window_bytes,
                    inst_bw_gib_s,
                    smooth_bw_gib_s,
                    avg_bw_gib_s,
                    force_done ? 1 : 0);
            fflush(progress_fp);
        }
        completed_since = 0;
        last_progress = now;
    };

    while (!g_stop)
    {
        poll_ready_slots();
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - t_start).count();
        if (args.seconds > 0 && elapsed_s >= args.seconds)
            break;
        if (args.total_bytes > 0 && completed_total >= args.total_bytes)
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
            // 1) 先对每个 stream 的当前 slot 做 gpuEventQuery（不阻塞）；
            // 2) 只有 slot 完成才提交下一批；
            // 3) 若本轮一个都提交不了，再阻塞等待一个 slot 完成，避免空转占满 CPU。
            bool issued_any = false;
            for (int i = 0; i < args.streams; ++i)
            {
                bool do_h2d = false, do_d2h = false;
                decide_dir(args, do_h2d, do_d2h);

                int k = ev_head[i];
                gpuError_t q = gpuEventQuery(done[i][k]);
                if (q == gpuErrorNotReady)
                    continue;
                if (q != gpuSuccess)
                    GPU_RT_CHECK(q);
                settle_slot(i, k);

                uint64_t slot_issued = 0;
                for (int bi = 0; bi < args.batch; ++bi)
                {
                    size_t copy_bytes = args.bytes;
                    if (args.total_bytes > 0)
                    {
                        if (issued_total >= args.total_bytes)
                            break;
                        uint64_t remaining = args.total_bytes - issued_total;
                        if ((uint64_t)copy_bytes > remaining)
                            copy_bytes = (size_t)remaining;
                    }
                    if (args.max_outstanding_bytes > 0)
                    {
                        uint64_t outstanding = issued_total - completed_total;
                        if (outstanding >= args.max_outstanding_bytes)
                            break;
                        uint64_t budget = args.max_outstanding_bytes - outstanding;
                        if ((uint64_t)copy_bytes > budget)
                            copy_bytes = (size_t)budget;
                    }
                    if (copy_bytes == 0)
                        break;

                    if (do_h2d)
                        GPU_RT_CHECK(gpuMemcpyAsync(dbufs[i], hbufs[i], copy_bytes, gpuMemcpyHostToDevice, streams[i]));
                    if (do_d2h)
                        GPU_RT_CHECK(gpuMemcpyAsync(hbufs[i], dbufs[i], copy_bytes, gpuMemcpyDeviceToHost, streams[i]));
                    slot_issued += (uint64_t)copy_bytes;
                    issued_total += (uint64_t)copy_bytes;
                }

                if (slot_issued == 0)
                {
                    if (args.total_bytes > 0 && issued_total >= args.total_bytes)
                        continue;
                    continue;
                }

                // 在该 slot 上记录“本批次结束”事件，供后续复用此 slot 前判断是否完成。
                GPU_RT_CHECK(gpuEventRecord(done[i][k], streams[i]));
                pending_bytes[i][k] = slot_issued;
                ev_head[i] = (k + 1) % args.inflight;
                issued_any = true;
            }

            if (!issued_any)
            {
                if (!wait_one_pending_slot())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        now = std::chrono::steady_clock::now();
        flush_progress(false);
    }

    // 退出前等待所有在途 memcpy 完成，保证统计与资源释放正确。
    GPU_RT_CHECK(gpuDeviceSynchronize());
    for (int i = 0; i < args.streams; ++i)
    {
        for (int k = 0; k < args.inflight; ++k)
            settle_slot(i, k);
    }
    flush_progress(true);

    // 最终汇总
    auto t_end = std::chrono::steady_clock::now();
    double wall_s = std::chrono::duration<double>(t_end - t_start).count();
    double avg_bw = (completed_total / (1024.0 * 1024.0 * 1024.0)) / (wall_s > 0 ? wall_s : 1e-9);
    printf("[pcie] done: wall=%.2fs issued_total=%lu completed_total=%lu avg_bw_gib_s=%.3f\n",
           wall_s, (unsigned long)issued_total, (unsigned long)completed_total, avg_bw);
    if (progress_fp)
        fclose(progress_fp);

    for (int i = 0; i < args.streams; ++i)
    {
        for (int k = 0; k < args.inflight; ++k)
            gpuEventDestroy(done[i][k]);

        if (args.pinned)
            gpuFreeHost(hbufs[i]);
        else
            free(hbufs[i]);
        gpuFree(dbufs[i]);
        gpuStreamDestroy(streams[i]);
    }
    return 0;
}
