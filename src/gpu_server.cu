// gpu_server.cu
// Build (A100/A800):
//   nvcc gpu_server.cu -O3 -std=c++14 -o gpu_server \
//        -I/usr/include/infiniband -L/usr/lib/x86_64-linux-gnu -libverbs -lpthread \
//        -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
//
// Usage:
//   ./gpu_server <ib_dev> <mr_size:{bytes|[0-9]+[KMG]}> <tcp_port> <port> <gid_idx> [mtu=1024]
//
// Behavior:
//   - The server keeps listening and can serve multiple clients sequentially.
//   - Stop it with Ctrl-C when you are done.
// Example:
//   ./gpu_server mlx5_1 1G 18515 1 3 1024

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <deque>
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

static void die(const char *m)
{
    perror(m);
    exit(1);
}

static void maybe_flush_gpudirect_writes()
{
    static bool supported = true;
    if (!supported)
        return;

#if !defined(CUDART_VERSION) || (CUDART_VERSION < 11030)
    supported = false;
    fprintf(stderr, "[server] cudaDeviceFlushGPUDirectRDMAWrites not available (CUDART_VERSION too old); ACK may not imply GPU visibility\n");
    return;
#else
    cudaError_t e = cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
                                                       cudaFlushGPUDirectRDMAWritesToOwner);
    if (e == cudaSuccess)
        return;
    if (e == cudaErrorNotSupported || e == cudaErrorInvalidValue)
    {
        // Clear the sticky error and disable subsequent flush attempts.
        cudaGetLastError();
        supported = false;
        fprintf(stderr, "[server] cudaDeviceFlushGPUDirectRDMAWrites not supported; ACK may not imply GPU visibility\n");
        return;
    }
    fprintf(stderr, "CUDA flush error: %s\n", cudaGetErrorString(e));
    exit(1);
#endif
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

struct ClientParams
{
    uint8_t op;        // 0=read, 1=write
    uint8_t write_ack; // 0/1
    uint16_t pad;
    uint32_t qd;
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
            die("qp INIT");
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
            die("qp RTR");
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
            die("qp RTS");
    }
}

static int make_listen_socket(int tcp_port)
{
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0)
        die("socket");
    int yes = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#ifdef SO_REUSEPORT
    setsockopt(ls, SOL_SOCKET, SO_REUSEPORT, &yes, sizeof(yes));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr *)&addr, sizeof(addr)) < 0)
        die("bind");
    if (listen(ls, 1) < 0)
        die("listen");
    return ls;
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s <ib_dev> <mr_size:{bytes|[0-9]+[KMG]}> <tcp_port> <port> <gid_idx> [mtu=1024]\n", argv[0]);
        return 1;
    }
    const char *devname = argv[1];
    size_t mr_bytes = parse_size(argv[2]);
    int tcp_port = atoi(argv[3]);
    uint8_t port = (uint8_t)atoi(argv[4]);
    int gid_idx = atoi(argv[5]);
    ibv_mtu mtu = (argc > 6 && atoi(argv[6]) == 2048) ? IBV_MTU_2048 : (argc > 6 && atoi(argv[6]) >= 4096 ? IBV_MTU_4096 : IBV_MTU_1024);

    int num_dev = 0;
    ibv_device **dl = ibv_get_device_list(&num_dev);
    if (!dl || !num_dev)
        die("no IB dev");
    ibv_device *ibdev = nullptr;
    for (int i = 0; i < num_dev; i++)
    {
        const char *n = ibv_get_device_name(dl[i]);
        if (n && std::string(n) == devname)
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
    // Create a CQ once and reuse it across connections.
    ibv_cq *cq = ibv_create_cq(ctx, 4096, nullptr, nullptr, 0);
    if (!cq)
        die("create_cq");

    // CUDA MR
    void *d_buf = nullptr;
    cudaError_t ce = cudaMalloc(&d_buf, mr_bytes);
    if (ce != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc %zu bytes failed: %s\n", mr_bytes, cudaGetErrorString(ce));
        return 2;
    }
    ibv_mr *mr = ibv_reg_mr(pd, d_buf, mr_bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!mr)
        die("ibv_reg_mr (GPU) (check nvidia-peermem)");

    fprintf(stderr, "[server] MR addr=0x%lx len=%zu rkey=0x%x\n", (unsigned long)(uintptr_t)d_buf, mr_bytes, mr->rkey);

    // Control channel: listen and serve clients sequentially.
    int ls = make_listen_socket(tcp_port);
    fprintf(stderr, "[server] listening on tcp_port=%d\n", tcp_port);

    while (true)
    {
        int s = accept(ls, nullptr, nullptr);
        if (s < 0)
            die("accept");
        int one = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        // Create a fresh QP per client so we can reconnect safely.
        ibv_qp_init_attr qia{};
        qia.qp_type = IBV_QPT_RC;
        qia.send_cq = cq;
        qia.recv_cq = cq;
        qia.cap.max_send_wr = 8192;
        qia.cap.max_recv_wr = 8192;
        qia.cap.max_send_sge = 1;
        qia.cap.max_recv_sge = 1;
        ibv_qp *qp = ibv_create_qp(pd, &qia);
        if (!qp)
            die("create_qp");

        ConnInfo local{}, remote{};
        local.qpn = qp->qp_num;
        local.port = port;
        local.gid_idx = (uint8_t)gid_idx;
        query_gid(ctx, port, gid_idx, local.gid);

        // Client sends first, then server replies.
        xr(s, &remote, sizeof(remote));
        xs(s, &local, sizeof(local));

        qp_to_rtr_rts_roce(qp, remote, port, gid_idx, mtu);

        struct
        {
            uint32_t rkey;
            uint64_t addr;
            uint64_t len;
        } rinfo{mr->rkey, (uint64_t)(uintptr_t)d_buf, (uint64_t)mr_bytes};
        xs(s, &rinfo, sizeof(rinfo));

        ClientParams params{};
        xr(s, &params, sizeof(params));
        bool need_ack = (params.op == 1 && params.write_ack);
        uint32_t qd = params.qd;
        if (need_ack)
        {
            fprintf(stderr, "[server] write_ack enabled: qd=%u (payload uses WRITE_WITH_IMM)\n", qd);
        }

        // signal ready so client can start issuing work
        char ready = 1;
        xs(s, &ready, 1);

        if (!need_ack)
        {
            // wait for client "done"
            char done{};
            xr(s, &done, 1);
            close(s);
            ibv_destroy_qp(qp);
            continue;
        }

        // Setup receive buffers for WRITE_WITH_IMM notifications.
        // In the new protocol, every payload WRITE_WITH_IMM generates one RECV-side CQE.
        // We keep recv_depth credits posted to avoid RNR when client pipelines writes.
        uint32_t recv_depth = qd > 0 ? qd : 1;
        if (recv_depth > 8192)
            recv_depth = 8192;
        std::vector<uint32_t> recv_bufs(recv_depth, 0);
        ibv_mr *recv_mr = ibv_reg_mr(pd, recv_bufs.data(), recv_bufs.size() * sizeof(uint32_t),
                                     IBV_ACCESS_LOCAL_WRITE);
        if (!recv_mr)
            die("ibv_reg_mr (recv)");
        std::vector<ibv_sge> recv_sges(recv_depth);
        std::vector<ibv_recv_wr> recv_wrs(recv_depth);
        for (uint32_t i = 0; i < recv_depth; i++)
        {
            recv_sges[i].addr = (uintptr_t)&recv_bufs[i];
            recv_sges[i].length = sizeof(uint32_t);
            recv_sges[i].lkey = recv_mr->lkey;
            recv_wrs[i] = {};
            recv_wrs[i].wr_id = i;
            recv_wrs[i].sg_list = &recv_sges[i];
            recv_wrs[i].num_sge = 1;
            recv_wrs[i].next = (i + 1 < recv_depth) ? &recv_wrs[i + 1] : nullptr;
        }
        ibv_recv_wr *bad_recv = nullptr;
        if (ibv_post_recv(qp, &recv_wrs[0], &bad_recv))
            die("ibv_post_recv");

        // ACK send buffer
        uint32_t *ack_buf = nullptr;
        if (posix_memalign((void **)&ack_buf, 64, sizeof(uint32_t)))
            die("posix_memalign");
        ibv_mr *ack_mr = ibv_reg_mr(pd, ack_buf, sizeof(uint32_t), IBV_ACCESS_LOCAL_WRITE);
        if (!ack_mr)
            die("ibv_reg_mr (ack)");

        ibv_sge ack_sge{};
        ibv_send_wr ack_wr{};
        ack_sge.addr = (uintptr_t)ack_buf;
        ack_sge.length = sizeof(uint32_t);
        ack_sge.lkey = ack_mr->lkey;
        ack_wr.sg_list = &ack_sge;
        ack_wr.num_sge = 1;
        ack_wr.opcode = IBV_WR_SEND;
        ack_wr.send_flags = IBV_SEND_SIGNALED;

        auto post_ack = [&](uint32_t tag) {
            // This is the visibility guarantee point:
            // only ACK after we force GPUDirect writes to become visible to the GPU.
            // If flush is unsupported, maybe_flush_gpudirect_writes() degrades once
            // and subsequent ACKs become best-effort (with warning already printed).
            maybe_flush_gpudirect_writes();
            *ack_buf = tag;
            ibv_send_wr *bad = nullptr;
            if (ibv_post_send(qp, &ack_wr, &bad))
                die("ibv_post_send (ack)");
        };

        // make TCP non-blocking so we can poll for "done"
        int flags = fcntl(s, F_GETFL, 0);
        if (flags >= 0)
            fcntl(s, F_SETFL, flags | O_NONBLOCK);

        bool done = false;
        bool ack_inflight = false;
        std::deque<uint32_t> pending;

        auto check_done = [&]() {
            if (done)
                return;
            char c = 0;
            ssize_t r = recv(s, &c, 1, MSG_DONTWAIT);
            if (r == 1 || r == 0)
                done = true;
            else if (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
                die("recv");
        };

        while (true)
        {
            ibv_wc wc{};
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0)
                die("poll_cq");
            if (n == 0)
            {
                check_done();
                if (done && !ack_inflight && pending.empty())
                    break;
                // Busy-poll for lowest ACK latency. This intentionally trades CPU usage
                // for reduced CQ idle-to-service delay.
                continue;
            }
            if (wc.status != IBV_WC_SUCCESS)
            {
                fprintf(stderr, "WC error: status=%d (%s), opcode=%d wr_id=%lu qpn=%u vend=0x%x\n",
                        wc.status, ibv_wc_status_str((ibv_wc_status)wc.status),
                        wc.opcode, wc.wr_id, wc.qp_num, wc.vendor_err);
                break;
            }
            if (wc.opcode == IBV_WC_RECV || wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM)
            {
                uint32_t tag = 0;
                if (wc.wc_flags & IBV_WC_WITH_IMM)
                    tag = ntohl(wc.imm_data);
                else
                    tag = recv_bufs[wc.wr_id];
                // Repost immediately so RQ credits remain stable under sustained traffic.
                recv_wrs[wc.wr_id].next = nullptr;
                if (ibv_post_recv(qp, &recv_wrs[wc.wr_id], &bad_recv))
                    die("ibv_post_recv (repost)");

                if (!ack_inflight)
                {
                    post_ack(tag);
                    ack_inflight = true;
                }
                else
                {
                    pending.push_back(tag);
                }
            }
            else if (wc.opcode == IBV_WC_SEND)
            {
                ack_inflight = false;
                if (!pending.empty())
                {
                    uint32_t tag = pending.front();
                    pending.pop_front();
                    post_ack(tag);
                    ack_inflight = true;
                }
            }
            else if (wc.opcode != IBV_WC_SEND)
            {
                fprintf(stderr, "[server] unexpected WC opcode=%d wr_id=%lu\n",
                        wc.opcode, wc.wr_id);
            }
            check_done();
            if (done && !ack_inflight && pending.empty())
                break;
        }

        ibv_dereg_mr(recv_mr);
        ibv_dereg_mr(ack_mr);
        free(ack_buf);
        close(s);
        ibv_destroy_qp(qp);
    }

    close(ls);
    return 0;
}
