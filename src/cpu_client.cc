// cpu_client.cc
// Build:
//   g++ cpu_client.cc -O3 -std=c++14 -o cpu_client \
//      -I/usr/include/infiniband -L/usr/lib/x86_64-linux-gnu -libverbs -lpthread
//
// Usage:
//   ./cpu_client <server_ip> <tcp_port> <ib_dev> <iters> <msg_bytes> <op:write|read> <port> <gid_idx> [qd=64] [span:{bytes|[0-9]+[KMG]}] [pattern:random|seq] [align=256] [mtu=1024] [sample=1] [max_samples=0] [ts_ms=0] [ts_out=] [write_ack=1] [read_min_qd=16] [rd_atomic=0(auto)] [qps=1]
// Example:
//   ./cpu_client 192.168.1.10 18515 mlx5_1 100000 65536 read 1 3 64 1G random 256 1024
//
// Notes:
//  - For read/write_no_ack, each QP keeps up to qd WRs in flight (qps controls QP count).
//  - Random offsets across 'span' (clamped to remote MR) make HBM activity visible.
//  - Uses ibv_wc_status_str for readable error logs.
//  - Optional runtime control is enabled via env vars:
//      SKYGDR_CTRL_PORT, SKYGDR_CTRL_HIGH_SLEEP_US, SKYGDR_CTRL_LOW_SLEEP_US, SKYGDR_CTRL_BIND_IP

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static void die(const char *m)
{
    perror(m);
    exit(1);
}
static void xs(int s, const void *b, size_t l)
{
    size_t o = 0;
    while (o < l)
    {
        ssize_t n = send(s, (char *)b + o, l - o, 0);
        if (n <= 0)
            die("send");
        o += n;
    }
}
static void xr(int s, void *b, size_t l)
{
    size_t o = 0;
    while (o < l)
    {
        ssize_t n = recv(s, (char *)b + o, l - o, MSG_WAITALL);
        if (n <= 0)
            die("recv");
        o += n;
    }
}

static size_t parse_size(const char *s)
{
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

// Get a monotonic timestamp in nanoseconds (steady clock avoids wall-clock jumps).
// We use this for per-op latency measurement.
static inline uint64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

enum ControlMode
{
    CONTROL_MODE_DISABLED = 0,
    CONTROL_MODE_HIGH = 1,
    CONTROL_MODE_LOW1 = 2,
    CONTROL_MODE_LOW2 = 3,
    CONTROL_MODE_LOW3 = 4,
    CONTROL_MODE_LOW4 = 5,
    CONTROL_MODE_CUSTOM = 6,
};

struct ClientControlState
{
    std::atomic<int> mode{CONTROL_MODE_DISABLED};
    std::atomic<uint32_t> pace_sleep_us{0};
    std::atomic<uint32_t> high_sleep_us{0};
    std::atomic<uint32_t> low1_sleep_us{0};
    std::atomic<uint32_t> low2_sleep_us{0};
    std::atomic<uint32_t> low3_sleep_us{0};
    std::atomic<uint32_t> low4_sleep_us{0};
    std::atomic<uint64_t> command_count{0};
    std::atomic<int> stop_requested{0};
    std::atomic<int> shutdown{0};
    uint16_t listen_port = 0;
    std::string bind_ip = "0.0.0.0";
};

static const char *control_mode_name(int mode)
{
    switch (mode)
    {
    case CONTROL_MODE_HIGH:
        return "HIGH";
    case CONTROL_MODE_LOW1:
        return "LOW1";
    case CONTROL_MODE_LOW2:
        return "LOW2";
    case CONTROL_MODE_LOW3:
        return "LOW3";
    case CONTROL_MODE_LOW4:
        return "LOW4";
    case CONTROL_MODE_CUSTOM:
        return "CUSTOM";
    default:
        return "DISABLED";
    }
}

static inline void apply_control_mode(ClientControlState &ctrl, int mode, uint32_t sleep_us)
{
    ctrl.mode.store(mode, std::memory_order_relaxed);
    ctrl.pace_sleep_us.store(sleep_us, std::memory_order_relaxed);
    ctrl.command_count.fetch_add(1, std::memory_order_relaxed);
}

static std::string trim_ascii(std::string s)
{
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
        s.pop_back();
    size_t pos = 0;
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t'))
        pos++;
    return s.substr(pos);
}

static void fill_control_response(const ClientControlState &ctrl, const char *prefix, std::string &resp)
{
    char buf[256];
    snprintf(buf, sizeof(buf),
             "%s mode=%s sleep_us=%u high_sleep_us=%u low1_sleep_us=%u low2_sleep_us=%u low3_sleep_us=%u low4_sleep_us=%u commands=%lu stop=%d\n",
             prefix,
             control_mode_name(ctrl.mode.load(std::memory_order_relaxed)),
             ctrl.pace_sleep_us.load(std::memory_order_relaxed),
             ctrl.high_sleep_us.load(std::memory_order_relaxed),
             ctrl.low1_sleep_us.load(std::memory_order_relaxed),
             ctrl.low2_sleep_us.load(std::memory_order_relaxed),
             ctrl.low3_sleep_us.load(std::memory_order_relaxed),
             ctrl.low4_sleep_us.load(std::memory_order_relaxed),
             (unsigned long)ctrl.command_count.load(std::memory_order_relaxed),
             ctrl.stop_requested.load(std::memory_order_relaxed));
    resp.assign(buf);
}

static void handle_control_command(const std::string &raw_cmd, ClientControlState &ctrl, std::string &resp)
{
    std::string cmd = trim_ascii(raw_cmd);
    if (cmd.empty())
    {
        fill_control_response(ctrl, "ERR empty", resp);
        return;
    }

    if (cmd == "HIGH")
    {
        apply_control_mode(ctrl, CONTROL_MODE_HIGH, ctrl.high_sleep_us.load(std::memory_order_relaxed));
        fill_control_response(ctrl, "OK", resp);
        return;
    }
    if (cmd == "LOW" || cmd == "LOW1")
    {
        apply_control_mode(ctrl, CONTROL_MODE_LOW1, ctrl.low1_sleep_us.load(std::memory_order_relaxed));
        fill_control_response(ctrl, "OK", resp);
        return;
    }
    if (cmd == "LOW2")
    {
        apply_control_mode(ctrl, CONTROL_MODE_LOW2, ctrl.low2_sleep_us.load(std::memory_order_relaxed));
        fill_control_response(ctrl, "OK", resp);
        return;
    }
    if (cmd == "LOW3")
    {
        apply_control_mode(ctrl, CONTROL_MODE_LOW3, ctrl.low3_sleep_us.load(std::memory_order_relaxed));
        fill_control_response(ctrl, "OK", resp);
        return;
    }
    if (cmd == "LOW4")
    {
        apply_control_mode(ctrl, CONTROL_MODE_LOW4, ctrl.low4_sleep_us.load(std::memory_order_relaxed));
        fill_control_response(ctrl, "OK", resp);
        return;
    }
    if (cmd == "STOP")
    {
        ctrl.stop_requested.store(1, std::memory_order_relaxed);
        ctrl.command_count.fetch_add(1, std::memory_order_relaxed);
        fill_control_response(ctrl, "OK", resp);
        return;
    }
    if (cmd == "STATUS")
    {
        fill_control_response(ctrl, "OK", resp);
        return;
    }
    if (cmd.rfind("SLEEP ", 0) == 0)
    {
        unsigned long v = strtoul(cmd.c_str() + 6, nullptr, 10);
        apply_control_mode(ctrl, CONTROL_MODE_CUSTOM, (uint32_t)v);
        fill_control_response(ctrl, "OK", resp);
        return;
    }

    fill_control_response(ctrl, "ERR unknown_cmd", resp);
}

static void control_server_loop(ClientControlState *ctrl)
{
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0)
    {
        perror("control socket");
        return;
    }
    int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    setsockopt(ls, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(ctrl->listen_port);
    if (ctrl->bind_ip.empty() || ctrl->bind_ip == "0.0.0.0")
    {
        addr.sin_addr.s_addr = INADDR_ANY;
    }
    else if (inet_pton(AF_INET, ctrl->bind_ip.c_str(), &addr.sin_addr) != 1)
    {
        fprintf(stderr, "[client] invalid control bind ip: %s\n", ctrl->bind_ip.c_str());
        close(ls);
        return;
    }

    if (bind(ls, (sockaddr *)&addr, sizeof(addr)) != 0)
    {
        perror("control bind");
        close(ls);
        return;
    }
    if (listen(ls, 4) != 0)
    {
        perror("control listen");
        close(ls);
        return;
    }

    fprintf(stderr, "[client] control listening on %s:%u\n", ctrl->bind_ip.c_str(), (unsigned)ctrl->listen_port);
    while (!ctrl->shutdown.load(std::memory_order_relaxed))
    {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(ls, &rfds);
        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 200000;
        int rc = select(ls + 1, &rfds, nullptr, nullptr, &tv);
        if (rc < 0)
        {
            perror("control select");
            break;
        }
        if (rc == 0)
            continue;

        int cs = accept(ls, nullptr, nullptr);
        if (cs < 0)
            continue;

        char buf[256];
        ssize_t n = recv(cs, buf, sizeof(buf) - 1, 0);
        if (n <= 0)
        {
            close(cs);
            continue;
        }
        buf[n] = '\0';

        std::string resp;
        handle_control_command(buf, *ctrl, resp);
        (void)send(cs, resp.data(), resp.size(), 0);
        close(cs);
    }

    close(ls);
}

struct ConnInfo
{
    uint32_t qpn;
    uint8_t gid[16];
    uint8_t port;
    uint8_t gid_idx;
    uint16_t pad;
};

static void query_gid(ibv_context *ctx, uint8_t port, int gid_idx, uint8_t out[16])
{
    ibv_gid g{};
    if (ibv_query_gid(ctx, port, gid_idx, &g))
        die("ibv_query_gid");
    memcpy(out, &g, 16);
}

static void qp_to_rtr_rts_roce(ibv_qp *qp, const ConnInfo &remote, uint8_t local_port, int local_gid_idx, ibv_mtu mtu, int rd_atomic)
{
    if (rd_atomic < 1)
        rd_atomic = 1;
    // INIT
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = local_port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            die("INIT");
    }
    // RTR
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = mtu;
        a.dest_qp_num = remote.qpn;
        a.rq_psn = 0;
        a.max_dest_rd_atomic = (uint8_t)rd_atomic;
        a.min_rnr_timer = 12;
        a.ah_attr.is_global = 1;
        a.ah_attr.port_num = local_port;
        a.ah_attr.grh.sgid_index = local_gid_idx;
        a.ah_attr.grh.hop_limit = 64;
        memcpy(&a.ah_attr.grh.dgid, remote.gid, 16);
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            die("RTR");
    }
    // RTS
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTS;
        a.timeout = 14;
        a.retry_cnt = 7;
        a.rnr_retry = 7;
        a.sq_psn = 0;
        a.max_rd_atomic = (uint8_t)rd_atomic;
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            die("RTS");
    }
}

// simple xorshift64 rng for random offsets
static inline uint64_t rng64(uint64_t &s)
{
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

int main(int argc, char **argv)
{
    if (argc < 10)
    {
        fprintf(stderr,
                "Usage: %s <server_ip> <tcp_port> <ib_dev> <iters> <msg_bytes> <op:write|read> <port> <gid_idx> [qd=64] [span:{bytes|[0-9]+[KMG]}] [pattern:random|seq] [align=256] [mtu=1024] [sample=1] [max_samples=0] [ts_ms=0] [ts_out=] [write_ack=1] [read_min_qd=16] [rd_atomic=0(auto)] [qps=1]\n",
                argv[0]);
        fprintf(stderr, "Env control: SKYGDR_CTRL_PORT / SKYGDR_CTRL_HIGH_SLEEP_US / SKYGDR_CTRL_LOW_SLEEP_US / SKYGDR_CTRL_LOW2_SLEEP_US / SKYGDR_CTRL_LOW3_SLEEP_US / SKYGDR_CTRL_LOW4_SLEEP_US / SKYGDR_CTRL_BIND_IP\n");
        return 1;
    }
    const char *sip = argv[1];
    int tcp_port = atoi(argv[2]);
    const char *devname = argv[3];
    uint64_t iters = strtoull(argv[4], nullptr, 10);
    size_t msg = parse_size(argv[5]);
    bool do_read = (strcmp(argv[6], "read") == 0);
    uint8_t port = (uint8_t)atoi(argv[7]);
    int gid_idx = atoi(argv[8]);

    int argi = 9;
    int qd = (argc > argi) ? atoi(argv[argi++]) : 64;
    size_t span = (argc > argi) ? parse_size(argv[argi++]) : 0; // 0 -> will be set to remote len later
    const char *pattern = (argc > argi) ? argv[argi++] : "random";
    size_t align = (argc > argi) ? (size_t)atoi(argv[argi++]) : 256;
    ibv_mtu mtu = IBV_MTU_1024;
    if (argc > argi)
    {
        int mtu_arg = atoi(argv[argi]);
        if (mtu_arg == 2048)
            mtu = IBV_MTU_2048;
        else if (mtu_arg >= 4096)
            mtu = IBV_MTU_4096;
        argi++;
    }
    // Latency sampling controls:
    //  - sample: take 1 out of N completions to reduce overhead (N>=1).
    //  - max_samples: cap total stored samples to bound memory (0 = no cap).
    int sample = (argc > argi) ? atoi(argv[argi++]) : 1;
    size_t max_samples = (argc > argi) ? (size_t)strtoull(argv[argi++], nullptr, 10) : 0;
    // Time-series controls (for aligning with GPU logger):
    //  - ts_ms: window size in ms (0 disables time-series output).
    //  - ts_out: CSV output path; "-" or empty means stdout.
    int ts_ms = (argc > argi) ? atoi(argv[argi++]) : 0;
    const char *ts_out = (argc > argi) ? argv[argi++] : nullptr;
    int write_ack = (argc > argi) ? atoi(argv[argi++]) : 1;
    int read_min_qd = 16;
    if (const char *e = getenv("SKYGDR_READ_MIN_QD"))
    {
        int v = atoi(e);
        if (v > 0)
            read_min_qd = v;
    }
    int rd_atomic_arg = 0;
    if (const char *e = getenv("SKYGDR_RD_ATOMIC"))
    {
        int v = atoi(e);
        if (v > 0)
            rd_atomic_arg = v;
    }
    int qps = 1;
    if (const char *e = getenv("SKYGDR_QPS"))
    {
        int v = atoi(e);
        if (v > 0)
            qps = v;
    }
    ClientControlState ctrl{};
    if (const char *e = getenv("SKYGDR_CTRL_PORT"))
    {
        int v = atoi(e);
        if (v > 0 && v <= 65535)
            ctrl.listen_port = (uint16_t)v;
    }
    if (const char *e = getenv("SKYGDR_CTRL_HIGH_SLEEP_US"))
    {
        unsigned long v = strtoul(e, nullptr, 10);
        ctrl.high_sleep_us.store((uint32_t)v, std::memory_order_relaxed);
    }
    if (const char *e = getenv("SKYGDR_CTRL_LOW_SLEEP_US"))
    {
        unsigned long v = strtoul(e, nullptr, 10);
        ctrl.low1_sleep_us.store((uint32_t)v, std::memory_order_relaxed);
    }
    if (const char *e = getenv("SKYGDR_CTRL_LOW2_SLEEP_US"))
    {
        unsigned long v = strtoul(e, nullptr, 10);
        ctrl.low2_sleep_us.store((uint32_t)v, std::memory_order_relaxed);
    }
    if (const char *e = getenv("SKYGDR_CTRL_LOW3_SLEEP_US"))
    {
        unsigned long v = strtoul(e, nullptr, 10);
        ctrl.low3_sleep_us.store((uint32_t)v, std::memory_order_relaxed);
    }
    if (const char *e = getenv("SKYGDR_CTRL_LOW4_SLEEP_US"))
    {
        unsigned long v = strtoul(e, nullptr, 10);
        ctrl.low4_sleep_us.store((uint32_t)v, std::memory_order_relaxed);
    }
    if (const char *e = getenv("SKYGDR_CTRL_BIND_IP"))
    {
        if (*e)
            ctrl.bind_ip = e;
    }
    // Optional tail compatibility:
    // - New format: [write_ack] [read_min_qd] [rd_atomic] [qps]
    // - Legacy one-field tail: [write_ack_batch] (deprecated; ignored)
    //   If there is only one remaining token and it is 0/1, treat it as legacy.
    int rem = argc - argi;
    if (rem == 1)
    {
        int v = atoi(argv[argi++]);
        if (v == 0 || v == 1)
        {
            if (v != 1)
                fprintf(stderr, "[client] write_ack_batch is deprecated and ignored; per-write ACK is always used\n");
        }
        else if (v > 0)
        {
            read_min_qd = v;
        }
    }
    else if (rem >= 2)
    {
        int v = atoi(argv[argi++]);
        if (v > 0)
            read_min_qd = v;
        v = atoi(argv[argi++]);
        if (v > 0)
            rd_atomic_arg = v;
        if (rem >= 3)
        {
            v = atoi(argv[argi++]);
            if (v > 0)
                qps = v;
        }
    }
    if (do_read)
    {
        write_ack = 0;
    }
    if (qps < 1)
        qps = 1;
    if (write_ack && qps > 1)
    {
        fprintf(stderr, "[client] write_ack path currently supports qps=1 only; forcing qps=1\n");
        qps = 1;
    }

    uint32_t low1_sleep_us = ctrl.low1_sleep_us.load(std::memory_order_relaxed);
    if (ctrl.low2_sleep_us.load(std::memory_order_relaxed) == 0 && low1_sleep_us > 0)
        ctrl.low2_sleep_us.store((uint32_t)((low1_sleep_us * 5U + 3U) / 4U), std::memory_order_relaxed);
    if (ctrl.low3_sleep_us.load(std::memory_order_relaxed) == 0 && low1_sleep_us > 0)
        ctrl.low3_sleep_us.store((uint32_t)((low1_sleep_us * 3U + 1U) / 2U), std::memory_order_relaxed);
    if (ctrl.low4_sleep_us.load(std::memory_order_relaxed) == 0 && low1_sleep_us > 0)
        ctrl.low4_sleep_us.store((uint32_t)((low1_sleep_us * 7U + 3U) / 4U), std::memory_order_relaxed);

    ctrl.mode.store(ctrl.listen_port > 0 ? CONTROL_MODE_HIGH : CONTROL_MODE_DISABLED, std::memory_order_relaxed);
    ctrl.pace_sleep_us.store(ctrl.high_sleep_us.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // TCP control
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0)
        die("socket");
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(tcp_port);
    if (inet_pton(AF_INET, sip, &a.sin_addr) != 1)
        die("inet_pton");
    if (connect(s, (sockaddr *)&a, sizeof(a)))
        die("connect");

    // RDMA device
    int nd = 0;
    ibv_device **dl = ibv_get_device_list(&nd);
    if (!dl || !nd)
        die("no IB dev");
    ibv_device *ibdev = nullptr;
    for (int i = 0; i < nd; i++)
    {
        const char *nm = ibv_get_device_name(dl[i]);
        if (nm && !strcmp(nm, devname))
        {
            ibdev = dl[i];
            break;
        }
    }
    if (!ibdev)
        die("dev not found");
    ibv_context *ctx = ibv_open_device(ibdev);
    if (!ctx)
        die("open_device");
    ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd)
        die("alloc_pd");

    if (qd < 1)
        qd = 1;
    if (do_read && qd < read_min_qd)
    {
        fprintf(stderr, "[client] read qd=%d is too shallow for throughput; auto-raise to %d (override by passing read_min_qd)\n",
                qd, read_min_qd);
        qd = read_min_qd;
    }
    int cq_depth = 8192;
    {
        uint64_t need = (uint64_t)qps * (uint64_t)qd * 4ULL;
        if (need > (uint64_t)cq_depth)
            cq_depth = (int)std::min<uint64_t>(need, 262144ULL);
    }
    ibv_cq *cq = ibv_create_cq(ctx, cq_depth, nullptr, nullptr, 0);
    if (!cq)
        die("create_cq");

    ibv_device_attr dev_attr{};
    if (ibv_query_device(ctx, &dev_attr))
        die("ibv_query_device");
    int rd_atomic_cap = std::min<int>(dev_attr.max_qp_rd_atom, dev_attr.max_qp_init_rd_atom);
    if (rd_atomic_cap < 1)
        rd_atomic_cap = 1;
    int rd_atomic = 1;
    if (do_read)
    {
        int desired = (rd_atomic_arg > 0) ? rd_atomic_arg : qd;
        if (desired < 1)
            desired = 1;
        rd_atomic = std::min(desired, rd_atomic_cap);
    }

    ibv_qp_init_attr qia{};
    qia.qp_type = IBV_QPT_RC;
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.cap.max_send_wr = 8192;
    qia.cap.max_recv_wr = (qd > 1) ? qd : 1;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    ibv_qp *qp0 = ibv_create_qp(pd, &qia);
    if (!qp0)
        die("create_qp");
    std::vector<ibv_qp *> qps_vec;
    qps_vec.reserve((size_t)qps);
    qps_vec.push_back(qp0);

    // Exchange conninfo (client sends first)
    ConnInfo local{}, remote{};
    local.qpn = qp0->qp_num;
    local.port = port;
    local.gid_idx = (uint8_t)gid_idx;
    query_gid(ctx, port, gid_idx, local.gid);
    xs(s, &local, sizeof(local));
    xr(s, &remote, sizeof(remote));

    // QP RTR/RTS
    qp_to_rtr_rts_roce(qp0, remote, port, gid_idx, mtu, rd_atomic);

    // Receive remote MR info
    struct
    {
        uint32_t rkey;
        uint64_t addr;
        uint64_t len;
    } rinfo{};
    xr(s, &rinfo, sizeof(rinfo));

    if (msg == 0 || msg > rinfo.len)
    {
        fprintf(stderr, "[client] invalid msg size vs remote len\n");
        return 2;
    }
    if (span == 0 || span > rinfo.len)
        span = rinfo.len;
    if (span < msg)
    {
        fprintf(stderr, "[client] span < msg; raising span to msg\n");
        span = msg;
    }
    fprintf(stderr, "[client] op=%s msg=%zu qd=%d qps=%d span=%zu align=%zu mtu=%d sample=%d max_samples=%zu ts_ms=%d ts_out=%s write_ack=%d read_min_qd=%d rd_atomic=%d(cap=%d)\n",
            do_read ? "READ" : "WRITE", msg, qd, qps, span, align, (mtu == IBV_MTU_4096 ? 4096 : mtu == IBV_MTU_2048 ? 2048
                                                                                                                     : 1024),
            sample, max_samples, ts_ms, ts_out ? ts_out : "-", write_ack, read_min_qd, rd_atomic, rd_atomic_cap);
    if (ctrl.listen_port > 0)
    {
        fprintf(stderr, "[client] control enabled: port=%u high_sleep_us=%u low1_sleep_us=%u low2_sleep_us=%u low3_sleep_us=%u low4_sleep_us=%u bind_ip=%s\n",
                (unsigned)ctrl.listen_port,
                ctrl.high_sleep_us.load(std::memory_order_relaxed),
                ctrl.low1_sleep_us.load(std::memory_order_relaxed),
                ctrl.low2_sleep_us.load(std::memory_order_relaxed),
                ctrl.low3_sleep_us.load(std::memory_order_relaxed),
                ctrl.low4_sleep_us.load(std::memory_order_relaxed),
                ctrl.bind_ip.c_str());
    }

    std::thread control_thread;

    // Local host buffer:
    // For multi-inflight one-sided operations (READ / WRITE without ACK), each WR should
    // use an independent local slot to avoid overlapping DMA writes on the same address.
    bool multi_inflight_data = (do_read || !write_ack);
    uint64_t io_slots64 = multi_inflight_data ? ((uint64_t)qd * (uint64_t)qps) : 1ULL;
    if (io_slots64 < 1)
        io_slots64 = 1;
    if (io_slots64 > (uint64_t)SIZE_MAX)
    {
        fprintf(stderr, "[client] io slot overflow: qd=%d qps=%d\n", qd, qps);
        return 2;
    }
    size_t io_slots = (size_t)io_slots64;
    size_t buf_bytes = msg;
    if (io_slots > 1)
    {
        if (msg > (SIZE_MAX / io_slots))
        {
            fprintf(stderr, "[client] local buffer size overflow: msg=%zu slots=%zu\n", msg, io_slots);
            return 2;
        }
        buf_bytes = msg * io_slots;
    }

    void *buf = nullptr;
    if (posix_memalign(&buf, 4096, buf_bytes))
        die("posix_memalign");
    if (!do_read)
    {
        // WRITE path only needs one initialized payload slot.
        memset(buf, 0xAB, msg);
    }
    int lflags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    ibv_mr *lmr = ibv_reg_mr(pd, buf, buf_bytes, lflags);
    if (!lmr)
        die("ibv_reg_mr (host)");

    // Exchange client params (so server knows whether to enable WRITE ack path).
    struct ClientParams
    {
        uint8_t op;        // 0=read, 1=write
        uint8_t write_ack; // 0/1
        uint16_t pad;
        uint32_t qd;
        uint32_t qps;
    } params{};
    params.op = do_read ? 0 : 1;
    params.write_ack = write_ack ? 1 : 0;
    params.qd = (uint32_t)qd;
    params.qps = (uint32_t)qps;
    xs(s, &params, sizeof(params));

    // Build extra data QPs if requested (client sends conninfo first per QP).
    for (int qi = 1; qi < qps; qi++)
    {
        ibv_qp *qpx = ibv_create_qp(pd, &qia);
        if (!qpx)
            die("create_qp");
        ConnInfo lx{}, rx{};
        lx.qpn = qpx->qp_num;
        lx.port = port;
        lx.gid_idx = (uint8_t)gid_idx;
        memcpy(lx.gid, local.gid, sizeof(lx.gid));
        xs(s, &lx, sizeof(lx));
        xr(s, &rx, sizeof(rx));
        qp_to_rtr_rts_roce(qpx, rx, port, gid_idx, mtu, rd_atomic);
        qps_vec.push_back(qpx);
    }

    char ready = 0;
    xr(s, &ready, 1);
    ibv_qp *qp = qp0;

    // Pre-build QD descriptors (we'll update remote_addr each post)
    std::vector<ibv_sge> sges(qd);
    std::vector<ibv_send_wr> wrs(qd);
    for (int i = 0; i < qd; i++)
    {
        int slot = multi_inflight_data ? i : 0;
        sges[i].addr = (uintptr_t)buf + (uintptr_t)((size_t)slot * msg);
        sges[i].length = msg;
        sges[i].lkey = lmr->lkey;
        wrs[i] = {};
        wrs[i].wr_id = i;
        wrs[i].sg_list = &sges[i];
        wrs[i].num_sge = 1;
        wrs[i].opcode = do_read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
        wrs[i].send_flags = IBV_SEND_SIGNALED;
        wrs[i].wr.rdma.rkey = rinfo.rkey;
    }

    const bool random_pattern = (strcmp(pattern, "random") == 0);
    uint64_t off = 0;
    uint64_t cur = 0;
    uint64_t seed = 0x9e3779b97f4a7c15ULL ^ (uint64_t)time(nullptr);

    auto next_off = [&](bool random) -> uint64_t
    {
        if (random)
        {
            uint64_t r = rng64(seed);
            uint64_t max_off = span - msg;
            if (max_off == 0)
                return 0;
            uint64_t o = r % (max_off + 1);
            o &= ~((uint64_t)align - 1ULL);
            if (o > max_off)
                o = max_off & ~((uint64_t)align - 1ULL);
            return o;
        }
        else
        {
            uint64_t o = cur;
            cur += msg;
            if (cur > span - msg)
                cur = 0;
            o &= ~((uint64_t)align - 1ULL);
            return o;
        }
    };

    // Time-series CSV (optional): output one CSV line per window (ts_ms).
    // This allows aligning RDMA latency with GPU metrics sampling.
    FILE *ts_fp = nullptr;
    if (ts_ms > 0)
    {
        if (ts_out && ts_out[0] && strcmp(ts_out, "-") != 0)
            ts_fp = fopen(ts_out, "w");
        else
            ts_fp = stdout;
        if (ts_fp)
        {
            fprintf(ts_fp, "ts_unix_ms,window_ms,window_us,ops,ops_per_s,throughput_gib_s,p50_us,p90_us,p99_us,p999_us,min_us,max_us,samples,control_mode,pace_sleep_us,control_commands\n");
            fflush(ts_fp);
        }
        else
        {
            fprintf(stderr, "[client] failed to open ts_out=%s\n", ts_out ? ts_out : "(null)");
        }
    }

    if (ctrl.listen_port > 0)
        control_thread = std::thread(control_server_loop, &ctrl);

    auto cleanup_and_exit = [&](int code) -> int {
        ctrl.shutdown.store(1, std::memory_order_relaxed);
        if (control_thread.joinable())
            control_thread.join();
        if (ts_fp && ts_fp != stdout)
        {
            fclose(ts_fp);
            ts_fp = nullptr;
        }
        return code;
    };

    // Main loop: post bursts of up to qd WRs, then poll those completions.
    // We record per-WR start timestamps (indexed by wr_id) and compute completion latency.
    uint64_t total_posted = 0, total_completed = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    uint64_t total_slots64 = (uint64_t)qps * (uint64_t)qd;
    if (total_slots64 < 1)
        total_slots64 = 1;
    if (total_slots64 > (uint64_t)SIZE_MAX)
    {
        fprintf(stderr, "[client] wr slot overflow: qps=%d qd=%d\n", qps, qd);
        return 2;
    }
    size_t total_wr_slots = (size_t)total_slots64;
    std::vector<uint64_t> start_ns(total_wr_slots, 0);
    // Store latencies in microseconds as double for higher precision.
    std::vector<double> lat_us;
    // Per-window latency samples for time-series output (also double).
    std::vector<double> lat_us_win;
    if (sample < 1)
        sample = 1;
    if (max_samples > 0)
    {
        uint64_t est = iters / (uint64_t)sample;
        if (est > max_samples)
            est = max_samples;
        lat_us.reserve((size_t)est);
    }
    else if (iters <= 1000000ULL)
    {
        lat_us.reserve((size_t)(iters / (uint64_t)sample + 1));
    }

    // Time-series state: track last report time and last completion count.
    auto last_report = std::chrono::steady_clock::now();
    uint64_t last_completed = 0;
    auto maybe_pace = [&]() {
        uint32_t sleep_us = ctrl.pace_sleep_us.load(std::memory_order_relaxed);
        if (sleep_us > 0)
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    };

    // Flush current window to CSV if time has elapsed or force=true at end.
    // We compute window ops/s, throughput, and latency percentiles.
    auto flush_window = [&](bool force) {
        if (ts_ms <= 0 || !ts_fp)
            return;
        auto now = std::chrono::steady_clock::now();
        auto dt_us = (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(now - last_report).count();
        if (!force && dt_us < (uint64_t)ts_ms * 1000ULL)
            return;
        uint64_t dt_ms = dt_us / 1000ULL;
        uint64_t ops_win = total_completed - last_completed;
        if (!force && ops_win == 0 && lat_us_win.empty())
            return;
        double sec_win = dt_us / 1000000.0;
        if (sec_win <= 0.0)
            sec_win = 1e-9;
        double ops_s = ops_win / sec_win;
        double gbps_win = ((double)msg * (double)ops_win) / (1024.0 * 1024.0 * 1024.0) / sec_win;

        double p50 = 0.0, p90 = 0.0, p99 = 0.0, p999 = 0.0, minv = 0.0, maxv = 0.0;
        if (!lat_us_win.empty())
        {
            std::sort(lat_us_win.begin(), lat_us_win.end());
            auto pct = [&](double p) -> double
            {
                if (lat_us_win.empty())
                    return 0.0;
                double idx = p * (double)(lat_us_win.size() - 1);
                size_t i0 = (size_t)idx;
                size_t i1 = (i0 + 1 < lat_us_win.size()) ? (i0 + 1) : i0;
                double frac = idx - (double)i0;
                return lat_us_win[i0] * (1.0 - frac) + lat_us_win[i1] * frac;
            };
            p50 = pct(0.50);
            p90 = pct(0.90);
            p99 = pct(0.99);
            p999 = pct(0.999);
            minv = lat_us_win.front();
            maxv = lat_us_win.back();
        }
        uint64_t ts_unix_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
        if (ops_win == 0 && lat_us_win.empty())
            return;
        fprintf(ts_fp, "%lu,%lu,%lu,%lu,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%zu,%s,%u,%lu\n",
                (unsigned long)ts_unix_ms,
                (unsigned long)dt_ms,
                (unsigned long)dt_us,
                (unsigned long)ops_win,
                ops_s, gbps_win,
                p50, p90, p99, p999,
                minv, maxv, lat_us_win.size(),
                control_mode_name(ctrl.mode.load(std::memory_order_relaxed)),
                ctrl.pace_sleep_us.load(std::memory_order_relaxed),
                (unsigned long)ctrl.command_count.load(std::memory_order_relaxed));
        fflush(ts_fp);
        lat_us_win.clear();
        last_report = now;
        last_completed = total_completed;
    };

    if (do_read || !write_ack)
    {
        // Keep QD steady by reposting immediately on completion.
        // This avoids burst drain/refill behavior that lowers effective in-flight depth.
        const int cq_poll_batch = 32;
        std::vector<ibv_wc> wcs((size_t)cq_poll_batch);

        const size_t total_slots = (size_t)qps * (size_t)qd;
        std::vector<std::vector<ibv_sge>> data_sges((size_t)qps, std::vector<ibv_sge>((size_t)qd));
        std::vector<std::vector<ibv_send_wr>> data_wrs((size_t)qps, std::vector<ibv_send_wr>((size_t)qd));
        std::vector<ibv_send_wr *> slot_wr(total_slots, nullptr);
        std::vector<int> slot_qp(total_slots, 0);
        for (int qi = 0; qi < qps; qi++)
        {
            for (int si = 0; si < qd; si++)
            {
                size_t sid = (size_t)qi * (size_t)qd + (size_t)si;
                int slot = multi_inflight_data ? (int)sid : 0;
                data_sges[(size_t)qi][(size_t)si].addr = (uintptr_t)buf + (uintptr_t)((size_t)slot * msg);
                data_sges[(size_t)qi][(size_t)si].length = msg;
                data_sges[(size_t)qi][(size_t)si].lkey = lmr->lkey;

                data_wrs[(size_t)qi][(size_t)si] = {};
                data_wrs[(size_t)qi][(size_t)si].wr_id = sid;
                data_wrs[(size_t)qi][(size_t)si].sg_list = &data_sges[(size_t)qi][(size_t)si];
                data_wrs[(size_t)qi][(size_t)si].num_sge = 1;
                data_wrs[(size_t)qi][(size_t)si].opcode = do_read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
                data_wrs[(size_t)qi][(size_t)si].send_flags = IBV_SEND_SIGNALED;
                data_wrs[(size_t)qi][(size_t)si].wr.rdma.rkey = rinfo.rkey;

                slot_wr[sid] = &data_wrs[(size_t)qi][(size_t)si];
                slot_qp[sid] = qi;
            }
        }

        auto queue_slot = [&](size_t sid, std::vector<ibv_send_wr *> &heads, std::vector<ibv_send_wr *> &tails) {
            int qi = slot_qp[sid];
            ibv_send_wr *wr = slot_wr[sid];
            off = next_off(random_pattern);
            wr->wr.rdma.remote_addr = rinfo.addr + off;
            wr->next = nullptr;
            start_ns[sid] = now_ns();
            if (!heads[(size_t)qi])
                heads[(size_t)qi] = wr;
            else
                tails[(size_t)qi]->next = wr;
            tails[(size_t)qi] = wr;
            total_posted++;
        };
        auto post_chains = [&](std::vector<ibv_send_wr *> &heads) -> bool {
            for (int qi = 0; qi < qps; qi++)
            {
                ibv_send_wr *head = heads[(size_t)qi];
                if (!head)
                    continue;
                ibv_send_wr *bad = nullptr;
                int post_rc = ibv_post_send(qps_vec[(size_t)qi], head, &bad);
                if (post_rc)
                {
                    fprintf(stderr, "ibv_post_send failed rc=%d (%s) qp=%d bad_wr_id=%lu bad_opcode=%d\n",
                            post_rc, strerror(post_rc), qi,
                            bad ? (unsigned long)bad->wr_id : 0UL,
                            bad ? (int)bad->opcode : -1);
                    return false;
                }
            }
            return true;
        };

        std::vector<ibv_send_wr *> init_heads((size_t)qps, nullptr);
        std::vector<ibv_send_wr *> init_tails((size_t)qps, nullptr);
        size_t initial = (size_t)std::min<uint64_t>((uint64_t)total_slots, iters - total_posted);
        for (size_t sid = 0; sid < initial; sid++)
            queue_slot(sid, init_heads, init_tails);
        if (!post_chains(init_heads))
            return cleanup_and_exit(3);
        if (initial > 0)
            maybe_pace();

        while (total_completed < total_posted || (!ctrl.stop_requested.load(std::memory_order_relaxed) && total_posted < iters))
        {
            int n = ibv_poll_cq(cq, cq_poll_batch, wcs.data());
            if (n < 0)
                die("poll_cq");
            if (n == 0)
            {
                if (ctrl.stop_requested.load(std::memory_order_relaxed) && total_completed >= total_posted)
                    break;
                continue;
            }

            std::vector<ibv_send_wr *> repost_heads((size_t)qps, nullptr);
            std::vector<ibv_send_wr *> repost_tails((size_t)qps, nullptr);
            for (int i = 0; i < n; i++)
            {
                ibv_wc &wc = wcs[(size_t)i];
                if (wc.status != IBV_WC_SUCCESS)
                {
                    fprintf(stderr, "WC error: status=%d (%s), opcode=%d wr_id=%lu qpn=%u vend=0x%x\n",
                            wc.status, ibv_wc_status_str((ibv_wc_status)wc.status),
                            wc.opcode, wc.wr_id, wc.qp_num, wc.vendor_err);
                    return cleanup_and_exit(4);
                }
                if (wc.wr_id >= (uint64_t)total_slots)
                {
                    fprintf(stderr, "WC error: unexpected wr_id=%lu (total_slots=%zu)\n",
                            (unsigned long)wc.wr_id, total_slots);
                    return cleanup_and_exit(4);
                }

                // Completion timestamp; latency is end-start for this wr_id.
                uint64_t end_ns = now_ns();
                double dur_us = (double)(end_ns - start_ns[(size_t)wc.wr_id]) / 1000.0;
                // Sample every N-th completion to reduce overhead.
                if ((total_completed % (uint64_t)sample) == 0)
                {
                    if (max_samples == 0 || lat_us.size() < max_samples)
                        lat_us.push_back(dur_us);
                    if (ts_ms > 0)
                        lat_us_win.push_back(dur_us);
                }
                total_completed++;
                // Emit time-series line when window ends.
                if (ts_ms > 0)
                    flush_window(false);

                if (!ctrl.stop_requested.load(std::memory_order_relaxed) && total_posted < iters)
                    queue_slot((size_t)wc.wr_id, repost_heads, repost_tails);
            }

            if (!post_chains(repost_heads))
                return cleanup_and_exit(3);
            bool posted_new = false;
            for (ibv_send_wr *head : repost_heads)
            {
                if (head)
                {
                    posted_new = true;
                    break;
                }
            }
            if (posted_new)
                maybe_pace();
        }
    }
    else
    {
        // WRITE with remote GPU-visibility ACK:
        //   - Each payload is sent as RDMA_WRITE_WITH_IMM(tag) directly to target offset.
        //   - Server receives RECV_RDMA_WITH_IMM, flushes GPUDirect writes to GPU visibility,
        //     and then SENDs a 4-byte ACK(tag) back.
        //   - Client treats ACK(tag) as the completion point for that payload's latency.
        //
        // Note:
        //   - This keeps one-step payload+notify semantics (no separate doorbell write).
        //   - ACK policy is strict per-write confirmation: at most one payload in flight.
        //     That gives the clearest "write became GPU-visible" completion semantics.
        const int max_inflight = 1;

        // Pre-post enough RECVs so server ACK SENDs can arrive without RNR.
        // Each RECV buffer stores one 4-byte ACK tag from server.
        std::vector<uint32_t> ack_tags(max_inflight, 0);
        ibv_mr *ack_mr = ibv_reg_mr(pd, ack_tags.data(),
                                    ack_tags.size() * sizeof(uint32_t),
                                    IBV_ACCESS_LOCAL_WRITE);
        if (!ack_mr)
            die("ibv_reg_mr (ack recv)");

        std::vector<ibv_sge> ack_sges(max_inflight);
        std::vector<ibv_recv_wr> ack_wrs(max_inflight);
        for (int i = 0; i < max_inflight; i++)
        {
            ack_sges[i] = {};
            ack_sges[i].addr = (uintptr_t)&ack_tags[i];
            ack_sges[i].length = sizeof(uint32_t);
            ack_sges[i].lkey = ack_mr->lkey;

            ack_wrs[i] = {};
            ack_wrs[i].wr_id = (uint64_t)i;
            ack_wrs[i].sg_list = &ack_sges[i];
            ack_wrs[i].num_sge = 1;
            ack_wrs[i].next = (i + 1 < max_inflight) ? &ack_wrs[i + 1] : nullptr;
        }
        ibv_recv_wr *bad_recv = nullptr;
        if (ibv_post_recv(qp, &ack_wrs[0], &bad_recv))
            die("ibv_post_recv (ack)");

        for (int i = 0; i < qd; i++)
        {
            wrs[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
            wrs[i].send_flags = 0;
            wrs[i].wr.rdma.rkey = rinfo.rkey;
        }

        uint32_t tag_seq = 1;
        while (total_completed < total_posted || (!ctrl.stop_requested.load(std::memory_order_relaxed) && total_posted < iters))
        {
            if (ctrl.stop_requested.load(std::memory_order_relaxed) && total_completed >= total_posted)
                break;
            int burst = (int)std::min<uint64_t>((uint64_t)max_inflight, iters - total_posted);
            if (burst <= 0)
                continue;
            std::unordered_map<uint32_t, int> tag_to_index;
            tag_to_index.reserve((size_t)burst * 2);

            for (int i = 0; i < burst; i++)
            {
                off = next_off(random_pattern);
                start_ns[i] = now_ns();
                wrs[i].wr.rdma.remote_addr = rinfo.addr + off;
                wrs[i].next = (i + 1 < burst) ? &wrs[i + 1] : nullptr;

                // Every payload WR carries its own immediate tag.
                uint32_t tag = tag_seq++;
                if (tag == 0)
                    tag = tag_seq++;
                tag_to_index[tag] = i;
                wrs[i].imm_data = htonl(tag);

                // Signal the tail WR so we can reclaim unsignaled WQEs.
                // This keeps SQ usage bounded while avoiding per-WR SEND CQE overhead.
                wrs[i].send_flags = (i + 1 < burst) ? 0 : IBV_SEND_SIGNALED;
                wrs[i].wr_id = 0x100000000ULL | (uint64_t)tag;
            }

            ibv_send_wr *bad = nullptr;
            int post_rc = ibv_post_send(qp, &wrs[0], &bad);
            if (post_rc)
            {
                fprintf(stderr, "ibv_post_send failed rc=%d (%s) bad_wr_id=%lu bad_opcode=%d\n",
                        post_rc, strerror(post_rc),
                        bad ? (unsigned long)bad->wr_id : 0UL,
                        bad ? (int)bad->opcode : -1);
                return cleanup_and_exit(3);
            }
            total_posted += burst;
            maybe_pace();

            bool got_send = false;
            std::vector<uint8_t> ack_seen((size_t)burst, 0);
            int acked = 0;
            while (!got_send || acked < burst)
            {
                ibv_wc wc{};
                int n = ibv_poll_cq(cq, 1, &wc);
                if (n < 0)
                    die("poll_cq");
                if (n == 0)
                    continue;
                if (wc.status != IBV_WC_SUCCESS)
                {
                    fprintf(stderr, "WC error: status=%d (%s), opcode=%d wr_id=%lu qpn=%u vend=0x%x\n",
                            wc.status, ibv_wc_status_str((ibv_wc_status)wc.status),
                            wc.opcode, wc.wr_id, wc.qp_num, wc.vendor_err);
                    return cleanup_and_exit(4);
                }
                if (wc.opcode == IBV_WC_RECV)
                {
                    if (wc.wr_id >= (uint64_t)ack_wrs.size())
                    {
                        fprintf(stderr, "[client] invalid ACK wr_id=%lu\n", wc.wr_id);
                        return cleanup_and_exit(4);
                    }

                    uint32_t ack_tag = ack_tags[(size_t)wc.wr_id];
                    auto it = tag_to_index.find(ack_tag);
                    if (it == tag_to_index.end())
                    {
                        fprintf(stderr, "[client] unexpected ACK tag=0x%x\n", ack_tag);
                        return cleanup_and_exit(4);
                    }
                    int idx = it->second;
                    if (ack_seen[(size_t)idx])
                    {
                        fprintf(stderr, "[client] duplicate ACK tag=0x%x\n", ack_tag);
                        return cleanup_and_exit(4);
                    }
                    ack_seen[(size_t)idx] = 1;
                    acked++;

                    uint64_t end_ns = now_ns();
                    double dur_us = (double)(end_ns - start_ns[(size_t)idx]) / 1000.0;
                    if ((total_completed % (uint64_t)sample) == 0)
                    {
                        if (max_samples == 0 || lat_us.size() < max_samples)
                            lat_us.push_back(dur_us);
                        if (ts_ms > 0)
                            lat_us_win.push_back(dur_us);
                    }
                    total_completed++;
                    if (ts_ms > 0)
                        flush_window(false);

                    // Repost this receive slot immediately so ACK credits stay stable.
                    ack_wrs[(size_t)wc.wr_id].next = nullptr;
                    if (ibv_post_recv(qp, &ack_wrs[(size_t)wc.wr_id], &bad_recv))
                        die("ibv_post_recv (ack repost)");
                }
                else if (wc.opcode == IBV_WC_RDMA_WRITE || wc.opcode == IBV_WC_SEND)
                {
                    // For RDMA_WRITE_WITH_IMM, providers typically report local SQ completion
                    // as IBV_WC_RDMA_WRITE (not IBV_WC_SEND). Accept both for portability.
                    got_send = true;
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ops = (double)total_completed / sec;
    double gbps = ((double)msg * (double)total_completed) / (1024.0 * 1024.0 * 1024.0) / sec;

    // Final flush of the last window (if time-series enabled).
    if (ts_ms > 0)
        flush_window(true);
    if (ts_fp && ts_fp != stdout)
    {
        fclose(ts_fp);
        ts_fp = nullptr;
    }

    printf("[client] done: iters=%lu msg=%zu qd=%d qps=%d span=%zu pattern=%s\n", iters, msg, qd, qps, span, pattern);
    printf("[client] elapsed=%.3f s  ops=%.0f ops/s  throughput=%.2f GiB/s\n", sec, ops, gbps);
    if (ctrl.stop_requested.load(std::memory_order_relaxed))
        printf("[client] control_stop=1 mode=%s pace_sleep_us=%u\n",
               control_mode_name(ctrl.mode.load(std::memory_order_relaxed)),
               ctrl.pace_sleep_us.load(std::memory_order_relaxed));
    if (!lat_us.empty())
    {
        std::sort(lat_us.begin(), lat_us.end());
        auto pct = [&](double p) -> double
        {
            if (lat_us.empty())
                return 0.0;
            double idx = p * (double)(lat_us.size() - 1);
            size_t i0 = (size_t)idx;
            size_t i1 = (i0 + 1 < lat_us.size()) ? (i0 + 1) : i0;
            double frac = idx - (double)i0;
            return lat_us[i0] * (1.0 - frac) + lat_us[i1] * frac;
        };
        double p50 = pct(0.50);
        double p90 = pct(0.90);
        double p99 = pct(0.99);
        double p999 = pct(0.999);
        double minv = lat_us.front();
        double maxv = lat_us.back();
        printf("[client] latency_us samples=%zu p50=%.3f p90=%.3f p99=%.3f p999=%.3f min=%.3f max=%.3f\n",
               lat_us.size(), p50, p90, p99, p999, minv, maxv);
    }
    else
    {
        printf("[client] latency_us samples=0\n");
    }

    // tell server we are done
    char done = 1;
    xs(s, &done, 1);
    ctrl.shutdown.store(1, std::memory_order_relaxed);
    if (control_thread.joinable())
        control_thread.join();
    return 0;
}
