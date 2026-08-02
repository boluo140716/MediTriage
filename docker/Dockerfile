FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_DEBUG=WARN
ENV MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST="8.0"
# HF 缓存钉到持久 bind 卷：否则按 HF-ID 加载的模型/数据集会落到容器临时写层 /root/.cache，
# 既与 /workspace/models、data/ 里的副本重复，又随容器重建丢失。
ENV HF_HOME=/workspace/data/hf_home
ENV HF_DATASETS_CACHE=/workspace/data/hf_cache

# System deps + Python 3.12 from deadsnakes PPA
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv \
    git wget curl ninja-build \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.10.0（与 vLLM 0.18.0 的依赖锁定一致，运行容器实测此组合；
# PyPI 轮自带 CUDA 运行时。勿钉更旧版本——会被下面 vLLM 层覆盖、白占镜像层）
RUN pip3 install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# vLLM 0.18.0 (veRL requires this exact version)
RUN pip3 install --no-cache-dir vllm==0.18.0

# flash-attn (build from source for this PyTorch+CUDA combo)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

# ML 工具链依赖（transformers / sentence-transformers / ray / wandb 等通用 ML 生态包）
RUN pip3 install --no-cache-dir \
    accelerate \
    transformers==4.57.6 \
    datasets \
    sentence-transformers \
    huggingface_hub \
    llamafactory \
    omegaconf \
    ray[default] \
    tensordict \
    torchdata \
    wandb \
    codetiming \
    liger-kernel \
    mathruler \
    peft \
    qwen-vl-utils \
    pylatexenc \
    requests \
    packaging

# Agent 运行时依赖，清单见 MediTriage/agent/requirements.txt
# （langgraph 编排 / pymilvus / 混合检索 / FastAPI 等；少了这层，
# 重建出的容器跑不起 Agent）。构建上下文须为仓库根：
#   docker build -f docker/Dockerfile -t medix .
COPY MediTriage/agent/requirements.txt /tmp/meditriage-requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/meditriage-requirements.txt \
    && rm /tmp/meditriage-requirements.txt

WORKDIR /workspace
