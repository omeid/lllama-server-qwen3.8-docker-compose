# ---------------------------------------------------------------------------
# Stage 1: llama.cpp (CUDA) + Hugging Face CLI on top of nvidia/cuda
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS llama-base

# Pin a llama.cpp release tag for reproducible builds, e.g. b6135
ARG LLAMA_CPP_REF=master
# Target arch only (86 = Ampere / RTX 3090 Ti) to keep the build fast and lean
ARG CUDAARCHS=86

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3 \
        python3-pip \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ggml-cuda records a DT_NEEDED on the driver lib `libcuda.so.1`, but the
# toolkit only ships the link stub `libcuda.so`. Alias it under the SONAME in
# a default linker search path so final executable links resolve it. At
# runtime the NVIDIA container toolkit mounts the real driver over this path.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so \
        /usr/lib/x86_64-linux-gnu/libcuda.so.1

RUN pip3 install --no-cache-dir -U "huggingface_hub[cli]"

WORKDIR /opt

RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp.git

RUN cmake -S llama.cpp -B llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} \
    && cmake --build llama.cpp/build --config Release -j"$(nproc)"

ENV PATH="/opt/llama.cpp/build/bin:$PATH"

# ---------------------------------------------------------------------------
# Stage 2: download the model into the Hugging Face cache
# ---------------------------------------------------------------------------
FROM llama-base AS model

ARG HF_SLUG
ARG HF_TOKEN

RUN hf download "${HF_SLUG%%:*}" --include "*${HF_SLUG#*:}*.gguf" --include "mmproj-BF16.gguf"
# ---------------------------------------------------------------------------
# Stage 3: runtime
# ---------------------------------------------------------------------------
FROM model AS server

ARG HF_SLUG
ENV HF_SLUG=${HF_SLUG}

ENV LLAMA_NGL=999 \
    LLAMA_CTX_SIZE=131072 \
    LLAMA_CACHE_TYPE_K=q8_0 \
    LLAMA_CACHE_TYPE_V=q8_0 \
    LLAMA_TEMP=0.75 \
    LLAMA_TOP_P=0.85 \
    LLAMA_TOP_K=40 \
    LLAMA_MIN_P=0.05 \
    LLAMA_PRESENCE_PENALTY=0.5 \
    LLAMA_NP=1 \
    LLAMA_CACHE_RAM=65536 \
    LLAMA_PORT=8033

CMD sh -c 'exec llama-server \
    -hf "${HF_SLUG}" \
    -ngl "${LLAMA_NGL}" \
    --ctx-size "${LLAMA_CTX_SIZE}" \
    --cache-type-k "${LLAMA_CACHE_TYPE_K}" \
    --cache-type-v "${LLAMA_CACHE_TYPE_V}" \
    --flash-attn on \
    --jinja \
    --temp "${LLAMA_TEMP}" \
    --top-p "${LLAMA_TOP_P}" \
    --top-k "${LLAMA_TOP_K}" \
    --min-p "${LLAMA_MIN_P}" \
    --presence-penalty "${LLAMA_PRESENCE_PENALTY}" \
    -np "${LLAMA_NP}" \
    --cache-ram "${LLAMA_CACHE_RAM}" \
    --cache-idle-slots \
    --host 0.0.0.0 \
    --port "${LLAMA_PORT}"'
