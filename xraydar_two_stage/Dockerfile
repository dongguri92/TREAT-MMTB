# TREAT-MMTB 2026 Task 1 offline inference container.
# PyTorch is installed from the CUDA 11.8 wheel index and carries its CUDA
# runtime; the host supplies only the NVIDIA driver through Container Toolkit.
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/workspace/.venv/bin:$PATH"

WORKDIR /workspace

COPY pyproject.toml uv.lock /workspace/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project \
    && uv pip check


FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/workspace/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=builder /workspace/.venv /workspace/.venv
COPY model.py predict.py window_normalization.py /workspace/
COPY weights/classifier_1.pt /workspace/weights/classifier_1.pt
COPY weights/classifier_2.pt /workspace/weights/classifier_2.pt
COPY weights/classifier_3.pt /workspace/weights/classifier_3.pt
COPY weights/unet.pt /workspace/weights/unet.pt
COPY weights/mask2former.pt /workspace/weights/mask2former.pt
COPY weights/window_normalizer.joblib /workspace/weights/window_normalizer.joblib
COPY runtime_config.json /workspace/weights/runtime_config.json

RUN test -s /workspace/weights/classifier_1.pt \
    && test -s /workspace/weights/classifier_2.pt \
    && test -s /workspace/weights/classifier_3.pt \
    && test -s /workspace/weights/unet.pt \
    && test -s /workspace/weights/mask2former.pt \
    && test -s /workspace/weights/window_normalizer.joblib \
    && test -s /workspace/weights/runtime_config.json \
    && python -c "from predict import load_models; import torch; load_models(['/workspace/weights/classifier_1.pt', '/workspace/weights/classifier_2.pt', '/workspace/weights/classifier_3.pt'], '/workspace/weights/unet.pt', '/workspace/weights/mask2former.pt', torch.device('cpu')); print('two-stage-ensemble-checkpoint-load-ok')"

ENTRYPOINT ["python", "/workspace/predict.py", \
            "--input", "/input", \
            "--output", "/output", \
            "--classifier-weights", "/workspace/weights/classifier_1.pt", "/workspace/weights/classifier_2.pt", "/workspace/weights/classifier_3.pt", \
            "--unet-weights", "/workspace/weights/unet.pt", \
            "--mask2former-weights", "/workspace/weights/mask2former.pt", \
            "--window-model", "/workspace/weights/window_normalizer.joblib", \
            "--runtime-config", "/workspace/weights/runtime_config.json"]
