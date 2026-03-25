#!/bin/bash
unset NCCL_NET
unset NCCL_NET_PLUGIN
unset NCCL_TUNER_CONFIG_PATH
unset NCCL_SOCKET_IFNAME
unset NCCL_NET_GDR_LEVEL
unset NCCL_CROSS_NIC
unset NCCL_IB_TC
unset NCCL_IB_FIFO_TC
unset NCCL_NVLS_CHUNKSIZE
unset NCCL_P2P_NET_CHUNKSIZE
unset NCCL_IB_QPS_PER_CONNECTION
unset NCCL_IB_ADAPTIVE_ROUTING

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

export PATH="$HOME/.local/bin:$PATH"
export UV_FIND_LINKS="$HOME/flash_attn_build/flash-attention/dist"
source "$HOME/skyrl_venv/bin/activate"

export WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_hf_token>
export HF_HUB_OFFLINE=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=100000
