./decode_transformer_kvcache_stress \
  --gpu 1 --layers 32 --kv-heads 8 --head-dim 128 \
  --context 16384 --gen 512 --ffn-dim 8192 \
  > decode_overlap_cached.log 2>&1 &
DECODE_PID=$!

./kv_nvlink_transfer_trace \
  --src 0 --dst 1 --dist cached \
  --layers 32 --kv-heads 8 --head-dim 128 --iters 1000 \
  > kv_overlap_cached.log 2>&1

wait $DECODE_PID

