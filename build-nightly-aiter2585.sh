#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-vllm-rocm-nightly-aiter2585}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai-rocm@sha256:263bceefd03baaf89a9e9a763f6ca7c0e4b8b2b0407d0666b699f169a1b55193}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${AITER_REF:-chuan/mla-nhead-lt16-support}"
GIT_IMAGE="${GIT_IMAGE:-alpine/git:latest}"
BUILD_DIR="${BUILD_DIR:-$(pwd)/.vllm-nightly-aiter2585-build}"
DOCKERFILE="${BUILD_DIR}/Dockerfile"

mkdir -p "${BUILD_DIR}"

cat >"${DOCKERFILE}" <<'DOCKERFILE'
ARG GIT_IMAGE=alpine/git:latest
ARG BASE_IMAGE=vllm/vllm-openai-rocm@sha256:263bceefd03baaf89a9e9a763f6ca7c0e4b8b2b0407d0666b699f169a1b55193
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=chuan/mla-nhead-lt16-support

FROM ${GIT_IMAGE} AS aiter_src
ARG AITER_REPO
ARG AITER_REF
WORKDIR /src
RUN git clone --depth 1 --branch "${AITER_REF}" "${AITER_REPO}" aiter

FROM ${BASE_IMAGE}

ENV VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_MLA=1

COPY --from=aiter_src /src/aiter/aiter/ /usr/local/lib/python3.12/dist-packages/aiter/
COPY --from=aiter_src /src/aiter/csrc/ /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/
COPY --from=aiter_src /src/aiter/gradlib/ /usr/local/lib/python3.12/dist-packages/aiter_meta/gradlib/
COPY --from=aiter_src /src/aiter/hsa/ /usr/local/lib/python3.12/dist-packages/aiter_meta/hsa/

RUN rm -f /usr/local/lib/python3.12/dist-packages/aiter/jit/module_mla_asm.so \
          /usr/local/lib/python3.12/dist-packages/aiter/jit/module_mla_metadata.so \
          /usr/local/lib/python3.12/dist-packages/aiter/jit/module_gemm_common.so && \
    rm -rf /usr/local/lib/python3.12/dist-packages/aiter/jit/build/module_mla_asm \
           /usr/local/lib/python3.12/dist-packages/aiter/jit/build/module_mla_metadata \
           /usr/local/lib/python3.12/dist-packages/aiter/jit/build/module_gemm_common
DOCKERFILE

echo "Building ${IMAGE_TAG}"
echo "  base:  ${BASE_IMAGE}"
echo "  AITER: ${AITER_REPO} @ ${AITER_REF}"
echo "  dockerfile: ${DOCKERFILE}"

docker build \
  "$@" \
  --pull \
  --build-arg "GIT_IMAGE=${GIT_IMAGE}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "AITER_REPO=${AITER_REPO}" \
  --build-arg "AITER_REF=${AITER_REF}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" \
  "${BUILD_DIR}"
