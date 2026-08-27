# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: compile llama.cpp with the CUDA backend
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS llama-build

# Pin a llama.cpp release tag for reproducible builds, e.g. b6135
ARG LLAMA_CPP_REF=master
# Target arch only (86 = Ampere / RTX 3090 Ti) to keep the build fast and lean
ARG CUDAARCHS=86

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

# ggml-cuda records a DT_NEEDED on the driver lib `libcuda.so.1`, but the
# toolkit only ships the link stub `libcuda.so`. Alias it under the SONAME in
# a default linker search path so final executable links resolve it. At
# runtime the NVIDIA container toolkit mounts the real driver over this path.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so \
        /usr/lib/x86_64-linux-gnu/libcuda.so.1

WORKDIR /opt

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp.git

RUN cmake -S llama.cpp -B llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} \
    && cmake --build llama.cpp/build --config Release -j"$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2: prepopulate the Hugging Face cache with the Q8 GGUF
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS model

ARG HF_REPO=unsloth/Qwen3.8-27B-GGUF
ARG HF_FILE_PATTERN=*Q8_0*.gguf

RUN pip install --no-cache-dir -U "huggingface_hub[cli]" \
    && hf download ${HF_REPO} --include "${HF_FILE_PATTERN}" \
        --cache-dir /root/.cache/huggingface

# Stable path for llama-server, pointing into the HF cache
RUN mkdir -p /models \
    && ln -s "$(find /root/.cache/huggingface -name '*.gguf' | head -n1)" \
        /models/Qwen3.8-27B-Q8_0.gguf

# ---------------------------------------------------------------------------
# Stage 3: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama-build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
# Prepopulated HF cache (so HF tooling resolves the model offline) + model
COPY --from=model /root/.cache/huggingface /root/.cache/huggingface
COPY --from=model /models /models

# 6x RTX 3090 Ti = 144 GiB VRAM total
#   Q8_0 weights            ~29 GiB
#   256k ctx, q8_0 KV cache ~32 GiB
#   -> fits with plenty of headroom for compute buffers
ENV MODEL="/models/Qwen3.8-27B-Q8_0.gguf" \
    HOST=0.0.0.0 \
    PORT=8080 \
    CTX_SIZE=256000 \
    N_GPU_LAYERS=999 \
    PARALLEL=4

EXPOSE 8080

CMD llama-server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --parallel "$PARALLEL" \
    --batch-size 2048 \
    --ubatch-size 512 \
    --jinja
