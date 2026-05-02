#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-vllm-v0151-aiter2585}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai-rocm:v0.15.1}"
AITER_REPO="${AITER_REPO:-https://github.com/ROCm/aiter.git}"
AITER_REF="${AITER_REF:-chuan/mla-nhead-lt16-support}"
GIT_IMAGE="${GIT_IMAGE:-alpine/git:latest}"
BUILD_DIR="${BUILD_DIR:-$(pwd)/.vllm-v0151-aiter2585-build}"
DOCKERFILE="${BUILD_DIR}/Dockerfile"
PATCHFILE="${BUILD_DIR}/patch_vllm_aiter_mla.py"

mkdir -p "${BUILD_DIR}"

cat >"${PATCHFILE}" <<'PY'
from pathlib import Path

path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
)
text = path.read_text()

text = text.replace("from typing import ClassVar\n", "from typing import ClassVar, Final\n")

helper = '''
class AiterMLAHelper:
    """Pad MLA query heads to AITER's 16-head minimum, then unpad output."""

    _AITER_MIN_MLA_HEADS: Final = 16

    @staticmethod
    def check_num_heads_validity(num_heads: int):
        assert AiterMLAHelper.is_valid_num_heads(num_heads), (
            f"Aiter MLA requires that num_heads be multiples or divisors of 16, "
            f"but provided {num_heads} number of heads.\\n"
            f"Try adjusting tensor_parallel_size value."
        )

    @staticmethod
    def is_valid_num_heads(num_heads: int) -> bool:
        return (
            num_heads % AiterMLAHelper._AITER_MIN_MLA_HEADS == 0
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else AiterMLAHelper._AITER_MIN_MLA_HEADS % num_heads == 0
        )

    @staticmethod
    def get_actual_mla_num_heads(num_heads: int) -> int:
        return max(num_heads, AiterMLAHelper._AITER_MIN_MLA_HEADS)

    @staticmethod
    def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
        return (
            q
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else q.repeat_interleave(
                AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, dim=1
            )
        )

    @staticmethod
    def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
        return (
            o
            if num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
            else o[:, :: AiterMLAHelper._AITER_MIN_MLA_HEADS // num_heads, :]
        )


'''

text = text.replace(
    "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):\n",
    helper + "class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):\n",
)
text = text.replace(
'''        assert num_heads == 16 or num_heads == 128, (
            f"Aiter MLA only supports 16 or 128 number of heads.\\n"
            f"Provided {num_heads} number of heads.\\n"
            "Try adjusting tensor_parallel_size value."
        )
''',
'''        AiterMLAHelper.check_num_heads_validity(num_heads)
''')
text = text.replace(
'''        B = q.shape[0]
        o = torch.zeros(
            B,
            self.num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )
''',
'''        B = q.shape[0]
        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self.num_heads)
        o = torch.zeros(
            B,
            mla_num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )
''')
text = text.replace(
'''        rocm_aiter_ops.mla_decode_fwd(
            q,
''',
'''        rocm_aiter_ops.mla_decode_fwd(
            mla_padded_q,
''')
text = text.replace(
'''        return o, None
''',
'''        return AiterMLAHelper.get_mla_unpadded_o(self.num_heads, o), None
''')

required = [
    "from typing import ClassVar, Final",
    "class AiterMLAHelper:",
    "AiterMLAHelper.check_num_heads_validity(num_heads)",
    "mla_padded_q = AiterMLAHelper.get_mla_padded_q",
    "return AiterMLAHelper.get_mla_unpadded_o",
]
missing = [marker for marker in required if marker not in text]
if missing:
    raise RuntimeError(f"Failed to patch rocm_aiter_mla.py; missing markers: {missing}")

path.write_text(text)
PY

cat >"${DOCKERFILE}" <<'DOCKERFILE'
ARG GIT_IMAGE=alpine/git:latest
ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.15.1
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
          /usr/local/lib/python3.12/dist-packages/aiter/jit/module_mla_metadata.so && \
    rm -rf /usr/local/lib/python3.12/dist-packages/aiter/jit/build/module_mla_asm \
           /usr/local/lib/python3.12/dist-packages/aiter/jit/build/module_mla_metadata

COPY patch_vllm_aiter_mla.py /tmp/patch_vllm_aiter_mla.py
RUN python3 /tmp/patch_vllm_aiter_mla.py && rm /tmp/patch_vllm_aiter_mla.py
DOCKERFILE

echo "Building ${IMAGE_TAG}"
echo "  base:  ${BASE_IMAGE}"
echo "  AITER: ${AITER_REPO} @ ${AITER_REF}"
echo "  dockerfile: ${DOCKERFILE}"

docker build \
  "$@" \
  --build-arg "GIT_IMAGE=${GIT_IMAGE}" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "AITER_REPO=${AITER_REPO}" \
  --build-arg "AITER_REF=${AITER_REF}" \
  -t "${IMAGE_TAG}" \
  -f "${DOCKERFILE}" \
  "${BUILD_DIR}"
