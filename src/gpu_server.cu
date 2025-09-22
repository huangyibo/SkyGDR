// gpu_server.cu
// Build: nvcc gpu_server.cu -O3 -std=c++14 -o gpu_server \
//        -I/usr/include/infiniband -L/usr/lib/x86_64-linux-gnu -libverbs -lpthread \
//        -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
//
// Usage: ./gpu_server <ib_dev> <msg_bytes> <tcp_port> <port> <gid_idx>
// Example: ./gpu_server mlx5_1 4096 18515 1 3
//
// Notes:
//  - Requires CUDA 11+/12+, NVIDIA driver, and nvidia-peermem loaded on the GPU host.
//  - RoCEv2: we use GIDs (no LIDs). QP AV uses global routing header (GRH).

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
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

struct ConnInfo
{
    uint32_t qpn;
    uint8_t gid[16];
    uint8_t port;    // e.g., 1
    uint8_t gid_idx; // e.g., 3
    uint16_t pad;    // align
};

static void query_gid(ibv_context *ctx, uint8_t port, int gid_idx, uint8_t out[16])
{
    ibv_gid g{};
    if (ibv_query_gid(ctx, port, gid_idx, &g))
        die("ibv_query_gid");
    memcpy(out, &g, 16);
}

static void qp_to_rtr_rts_roce(ibv_qp *qp, const ConnInfo &remote, uint8_t local_port, int local_gid_idx)
{
    // INIT
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = local_port;
        a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            die("modify_qp INIT");
    }
    // RTR
    {
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = IBV_MTU_1024; // safer default for typical RoCE PMTU; raise if fabric supports
        a.dest_qp_num = remote.qpn;
        a.rq_psn = 0;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer = 12;

        a.ah_attr.is_global = 1;
        a.ah_attr.port_num = local_port;
        a.ah_attr.grh.hop_limit = 64;
        a.ah_attr.grh.sgid_index = local_gid_idx;
        memcpy(&a.ah_attr.grh.dgid, remote.gid, 16);

        if (ibv_modify_qp(qp, &a,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                              IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            die("modify_qp RTR");
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
        if (ibv_modify_qp(qp, &a,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            die("modify_qp RTS");
    }
}

int main(int argc, char **argv)
{
    if (argc < 6)
    {
        fprintf(stderr, "Usage: %s <ib_dev> <msg_bytes> <tcp_port> <port> <gid_idx>\n", argv[0]);
        return 1;
    }
    const char *devname = argv[1];
    size_t msg_bytes = (size_t)atoll(argv[2]);
    int tcp_port = atoi(argv[3]);
    uint8_t port = (uint8_t)atoi(argv[4]);
    int gid_idx = atoi(argv[5]);

    // Open IB device by exact name
    int num_dev = 0;
    ibv_device **dev_list = ibv_get_device_list(&num_dev);
    if (!dev_list || !num_dev)
        die("no IB devices");
    ibv_device *ibdev = nullptr;
    for (int i = 0; i < num_dev; i++)
    {
        const char *n = ibv_get_device_name(dev_list[i]);
        if (n && std::string(n) == devname)
        {
            ibdev = dev_list[i];
            break;
        }
    }
    if (!ibdev)
        die("IB dev not found");
    ibv_context *ctx = ibv_open_device(ibdev);
    if (!ctx)
        die("open_device");
    ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd)
        die("alloc_pd");
    ibv_cq *cq = ibv_create_cq(ctx, 4096, nullptr, nullptr, 0);
    if (!cq)
        die("create_cq");

    // Create QP
    ibv_qp_init_attr qia{};
    qia.qp_type = IBV_QPT_RC;
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.cap.max_send_wr = 1024;
    qia.cap.max_recv_wr = 1024;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    ibv_qp *qp = ibv_create_qp(pd, &qia);
    if (!qp)
        die("create_qp");

    // CUDA buffer + MR registration (requires nvidia-peermem)
    void *d_buf = nullptr;
    cudaError_t ce = cudaMalloc(&d_buf, msg_bytes);
    if (ce != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(ce));
        return 2;
    }
    ibv_mr *mr = ibv_reg_mr(pd, d_buf, msg_bytes,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!mr)
        die("ibv_reg_mr (GPU) failed; is nvidia-peermem loaded?");

    // TCP control channel (listen)
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0)
        die("socket");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(tcp_port);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(ls, (sockaddr *)&addr, sizeof(addr)))
        die("bind");
    if (listen(ls, 1))
        die("listen");
    int s = accept(ls, nullptr, nullptr);
    if (s < 0)
        die("accept");

    // Exchange conn info with client (client sends first, server receives then replies)
    ConnInfo local{}, remote{};
    local.qpn = qp->qp_num;
    local.port = port;
    local.gid_idx = (uint8_t)gid_idx;
    query_gid(ctx, port, gid_idx, local.gid);

    xr(s, &remote, sizeof(remote));
    xs(s, &local, sizeof(local));

    // Move QP to RTR/RTS
    qp_to_rtr_rts_roce(qp, remote, port, gid_idx);

    // Send memory rkey/addr so client can RDMA READ/WRITE
    struct
    {
        uint32_t rkey;
        uint64_t addr;
        uint32_t len;
    } rinfo{mr->rkey, (uint64_t)(uintptr_t)d_buf, (uint32_t)msg_bytes};
    xs(s, &rinfo, sizeof(rinfo));

    // Keep server alive until client sends a one-byte "done"
    char done{};
    xr(s, &done, 1);
    close(s);
    close(ls);

    return 0;
}
