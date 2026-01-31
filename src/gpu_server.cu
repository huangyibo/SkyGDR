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

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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
        qia.cap.max_send_wr = 4096;
        qia.cap.max_recv_wr = 1;
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

        // wait for client "done"
        char done{};
        xr(s, &done, 1);
        close(s);
        ibv_destroy_qp(qp);
    }

    close(ls);
    return 0;
}
