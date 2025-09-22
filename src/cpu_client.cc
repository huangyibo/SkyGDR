// cpu_client.cc
// Build: g++ cpu_client.cc -O3 -std=c++14 -o cpu_client \
//        -I/usr/include/infiniband -L/usr/lib/x86_64-linux-gnu -libverbs -lpthread
//
// Usage: ./cpu_client <server_ip> <tcp_port> <ib_dev> <iters> <bytes> <op> <port> <gid_idx>
//   op: write | read
// Example:
//   ./cpu_client 192.168.1.10 18515 mlx5_1 10000 4096 write 1 3

#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netinet/in.h>
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

struct ConnInfo
{
    uint32_t qpn;
    uint8_t gid[16];
    uint8_t port;    // e.g., 1
    uint8_t gid_idx; // e.g., 3
    uint16_t pad;
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
        a.path_mtu = IBV_MTU_1024; // safer default; bump if your RoCE PMTU allows
        a.dest_qp_num = remote.qpn;
        a.rq_psn = 0;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer = 12;

        a.ah_attr.is_global = 1;
        a.ah_attr.port_num = local_port;
        a.ah_attr.grh.sgid_index = local_gid_idx;
        a.ah_attr.grh.hop_limit = 64;
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
    if (argc < 9)
    {
        fprintf(stderr,
                "Usage: %s <server_ip> <tcp_port> <ib_dev> <iters> <bytes> <op=write|read> <port> <gid_idx>\n", argv[0]);
        return 1;
    }
    const char *sip = argv[1];
    int tcp_port = atoi(argv[2]);
    const char *devname = argv[3];
    int iters = atoi(argv[4]);
    size_t msg = (size_t)atoll(argv[5]);
    bool do_read = (strcmp(argv[6], "read") == 0);
    uint8_t port = (uint8_t)atoi(argv[7]);
    int gid_idx = atoi(argv[8]);

    // TCP ctrl: connect to server
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0)
        die("socket");
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(tcp_port);
    if (inet_pton(AF_INET, sip, &a.sin_addr) != 1)
        die("inet_pton");
    if (connect(s, (sockaddr *)&a, sizeof(a)))
        die("connect");

    // Open RDMA dev by name
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
    ibv_cq *cq = ibv_create_cq(ctx, 1024, nullptr, nullptr, 0);
    if (!cq)
        die("create_cq");

    ibv_qp_init_attr qia{};
    qia.qp_type = IBV_QPT_RC;
    qia.send_cq = cq;
    qia.recv_cq = cq;
    qia.cap.max_send_wr = 1024;
    qia.cap.max_recv_wr = 1;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    ibv_qp *qp = ibv_create_qp(pd, &qia);
    if (!qp)
        die("create_qp");

    // Build/send our local conn info, then receive server's
    ConnInfo local{}, remote{};
    local.qpn = qp->qp_num;
    local.port = port;
    local.gid_idx = (uint8_t)gid_idx;
    query_gid(ctx, port, gid_idx, local.gid);

    xs(s, &local, sizeof(local));
    xr(s, &remote, sizeof(remote));

    // QP transitions
    qp_to_rtr_rts_roce(qp, remote, port, gid_idx);

    // Receive server memory info (GPU MR)
    struct
    {
        uint32_t rkey;
        uint64_t addr;
        uint32_t len;
    } rinfo{};
    xr(s, &rinfo, sizeof(rinfo));
    if (msg > rinfo.len)
        msg = rinfo.len;

    // Register a local host buffer
    void *buf = nullptr;
    if (posix_memalign(&buf, 4096, msg))
        die("posix_memalign");
    memset(buf, 0xAB, msg);
    ibv_mr *lmr = ibv_reg_mr(pd, buf, msg, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!lmr)
        die("ibv_reg_mr (host)");

    // Prepare WR template
    ibv_sge sge{};
    sge.addr = (uintptr_t)buf;
    sge.length = msg;
    sge.lkey = lmr->lkey;
    ibv_send_wr wr{};
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.opcode = do_read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
    wr.wr.rdma.remote_addr = rinfo.addr;
    wr.wr.rdma.rkey = rinfo.rkey;

    // Measure one-way post->CQE latency
    std::vector<double> us;
    us.reserve(iters);
    for (int i = 0; i < iters; i++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        ibv_send_wr *bad = nullptr;
        if (ibv_post_send(qp, &wr, &bad))
            die("ibv_post_send");
        ibv_wc wc{};
        while (true)
        {
            int n = ibv_poll_cq(cq, 1, &wc);
            if (n < 0)
                die("poll_cq");
            if (n == 0)
                continue;
            if (wc.status != IBV_WC_SUCCESS)
            {
                fprintf(stderr, "WC status=%d\n", wc.status);
                die("wc");
            }
            break;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    auto pct = [&](double p)
    {
        auto v = us;
        std::sort(v.begin(), v.end());
        if (v.empty())
            return 0.0;
        size_t idx = (size_t)((p / 100.0) * (v.size() - 1));
        return v[idx];
    };
    double mean = 0;
    for (double x : us)
        mean += x;
    mean /= us.size();

    printf("One-way %s to remote GPU buf (bytes=%zu, iters=%d): mean=%.2f us P50=%.2f P95=%.2f P99=%.2f P99.9=%.2f\n",
           do_read ? "READ" : "WRITE", msg, iters, mean, pct(50), pct(95), pct(99), pct(99.9));

    // Write CSV
    FILE *f = fopen("latency_us.csv", "w");
    if (f)
    {
        fprintf(f, "iter,latency_us\n");
        for (size_t i = 0; i < us.size(); ++i)
            fprintf(f, "%zu,%.3f\n", i, us[i]);
        fclose(f);
    }

    // Tell server we're done
    char done = 1;
    xs(s, &done, 1);
    return 0;
}
