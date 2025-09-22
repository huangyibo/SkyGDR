// cpu_client.cc
// Build:
//   g++ cpu_client.cc -O3 -std=c++14 -o cpu_client \
//      -I/usr/include/infiniband -L/usr/lib/x86_64-linux-gnu -libverbs -lpthread
//
// Usage:
//   ./cpu_client <server_ip> <tcp_port> <ib_dev> <iters> <msg_bytes> <op:write|read> <port> <gid_idx> [qd=64] [span:{bytes|[0-9]+[KMG]}] [pattern:random|seq] [align=256] [mtu=1024]
// Example:
//   ./cpu_client 192.168.1.10 18515 mlx5_1 100000 65536 read 1 3 64 1G random 256 1024
//
// Notes:
//  - With qd>1 we post a burst and poll qd completions in a loop.
//  - Random offsets across 'span' (clamped to remote MR) make HBM activity visible.
//  - Uses ibv_wc_status_str for readable error logs.

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

static void qp_to_rtr_rts_roce(ibv_qp *qp, const ConnInfo &remote, uint8_t local_port, int local_gid_idx, ibv_mtu mtu)
{
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
        a.max_dest_rd_atomic = 1;
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
        a.max_rd_atomic = 1;
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
                "Usage: %s <server_ip> <tcp_port> <ib_dev> <iters> <msg_bytes> <op:write|read> <port> <gid_idx> [qd=64] [span:{bytes|[0-9]+[KMG]}] [pattern:random|seq] [align=256] [mtu=1024]\n",
                argv[0]);
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
    ibv_mtu mtu = (argc > argi && atoi(argv[argi]) == 2048) ? IBV_MTU_2048 : (argc > argi && atoi(argv[argi]) >= 4096 ? IBV_MTU_4096 : IBV_MTU_1024);

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
    ibv_cq *cq = ibv_create_cq(ctx, 8192, nullptr, nullptr, 0);
    if (!cq)
        die("create_cq");

    ibv_qp_init_attr qia{};
    qia.qp_type = IBV_QPT_RC;
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.cap.max_send_wr = 8192;
    qia.cap.max_recv_wr = 1;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    ibv_qp *qp = ibv_create_qp(pd, &qia);
    if (!qp)
        die("create_qp");

    // Exchange conninfo (client sends first)
    ConnInfo local{}, remote{};
    local.qpn = qp->qp_num;
    local.port = port;
    local.gid_idx = (uint8_t)gid_idx;
    query_gid(ctx, port, gid_idx, local.gid);
    xs(s, &local, sizeof(local));
    xr(s, &remote, sizeof(remote));

    // QP RTR/RTS
    qp_to_rtr_rts_roce(qp, remote, port, gid_idx, mtu);

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
    if (qd < 1)
        qd = 1;

    fprintf(stderr, "[client] op=%s msg=%zu qd=%d span=%zu align=%zu mtu=%d\n",
            do_read ? "READ" : "WRITE", msg, qd, span, align, (mtu == IBV_MTU_4096 ? 4096 : mtu == IBV_MTU_2048 ? 2048
                                                                                                                : 1024));

    // Local host buffer (we can reuse one buffer for all WRs)
    void *buf = nullptr;
    if (posix_memalign(&buf, 4096, msg))
        die("posix_memalign");
    memset(buf, 0xAB, msg);
    int lflags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    ibv_mr *lmr = ibv_reg_mr(pd, buf, msg, lflags);
    if (!lmr)
        die("ibv_reg_mr (host)");

    // Pre-build QD descriptors (we'll update remote_addr each post)
    std::vector<ibv_sge> sges(qd);
    std::vector<ibv_send_wr> wrs(qd);
    for (int i = 0; i < qd; i++)
    {
        sges[i].addr = (uintptr_t)buf;
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

    // Main loop: post bursts of up to qd WRs, then poll those completions
    uint64_t total_posted = 0, total_completed = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    while (total_posted < iters)
    {
        int burst = (int)std::min<uint64_t>(qd, iters - total_posted);
        for (int i = 0; i < burst; i++)
        {
            off = next_off(std::string(pattern) == "random");
            wrs[i].wr.rdma.remote_addr = rinfo.addr + off;
            wrs[i].next = (i + 1 < burst) ? &wrs[i + 1] : nullptr;
        }
        ibv_send_wr *bad = nullptr;
        if (ibv_post_send(qp, &wrs[0], &bad))
        {
            perror("ibv_post_send");
            return 3;
        }
        total_posted += burst;

        // poll 'burst' completions
        int comp = 0;
        while (comp < burst)
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
                return 4;
            }
            comp++;
        }
        total_completed += burst;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ops = (double)total_completed / sec;
    double gbps = ((double)msg * (double)total_completed) / (1024.0 * 1024.0 * 1024.0) / sec;

    printf("[client] done: iters=%lu msg=%zu qd=%d span=%zu pattern=%s\n", iters, msg, qd, span, pattern);
    printf("[client] elapsed=%.3f s  ops=%.0f ops/s  throughput=%.2f GiB/s\n", sec, ops, gbps);

    // tell server we are done
    char done = 1;
    xs(s, &done, 1);
    return 0;
}
