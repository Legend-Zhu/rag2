#!/bin/bash
# E0 完整跑 - 双模型启动脚本
# 用法:
#   ./run_e0.sh qwen   # 跑 Qwen3.8 n=100
#   ./run_e0.sh k3     # 跑 K3 n=100
#   ./run_e0.sh both   # 顺序跑两个（K3 等 Qwen3.8 完）

set -e
cd /Users/pfpman/ZCodeProject

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export PYTHONPATH=src

MODEL=$1
N=${2:-100}

case "$MODEL" in
  qwen)
    : "${QWEN_API_KEY:?QWEN_API_KEY must be set in env (no longer hardcoded)}"
    echo "[$(date)] 启动 Qwen3.8 E0 n=$N ..."
    nohup python3 src/rag2/experiments/e0_full.py --model qwen3.8 --n $N \
      > logs/e0_qwen_n$N.log 2>&1 &
    echo "PID: $!  日志: logs/e0_qwen_n$N.log"
    ;;
  k3)
    : "${KIMI_API_KEY:?KIMI_API_KEY must be set in env (no longer hardcoded)}"
    echo "[$(date)] 启动 K3 E0 n=$N ..."
    nohup python3 src/rag2/experiments/e0_full.py --model kimi-k3 --n $N \
      > logs/e0_k3_n$N.log 2>&1 &
    echo "PID: $!  日志: logs/e0_k3_n$N.log"
    ;;
  both)
    : "${QWEN_API_KEY:?QWEN_API_KEY must be set in env (no longer hardcoded)}"
    : "${KIMI_API_KEY:?KIMI_API_KEY must be set in env (no longer hardcoded)}"
    echo "[$(date)] 顺序跑双模型（Qwen 先，K3 后）..."
    python3 src/rag2/experiments/e0_full.py --model qwen3.8 --n $N \
      > logs/e0_qwen_n$N.log 2>&1
    echo "[$(date)] Qwen3.8 完成，启动 K3 ..."
    python3 src/rag2/experiments/e0_full.py --model kimi-k3 --n $N \
      > logs/e0_k3_n$N.log 2>&1
    echo "[$(date)] 双模型 E0 全部完成"
    ;;
  *)
    echo "用法: $0 {qwen|k3|both} [n=100]"
    exit 1
    ;;
esac
