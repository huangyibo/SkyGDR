mkdir -p overlap_multi_kv

./decode_hbm_stress 1 32768 1000 > overlap_multi_kv/hbm_overlap.log 2>&1 &
HBM_PID=$!

sleep 1

for i in 0 1 2 3; do
  ./kv_nvlink_transfer 0 1 8192 1000 > overlap_multi_kv/kv_${i}.log 2>&1 &
done

wait
