# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY s2_organisation_and_version_control/s2/ s2_organisation_and_version_control/s2/
COPY data/ data/

WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENTRYPOINT ["uv", "run", "s2_organisation_and_version_control/s2/src/s2_exercises/train.py"]

# docker build -f train.dockerfile . -t train:latest
# docker run --rm --name experiment4 -v /Users/carlahugod/Desktop/UNI/7sem/MLOps/dtu_mlops/s2_organisation_and_version_control/s2/models:/s2_organisation_and_version_control/s2/models/ train:latest 