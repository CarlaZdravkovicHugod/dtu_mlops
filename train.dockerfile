# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY s2_organisation_and_version_control/s2/src/ s2_organisation_and_version_control/s2/src/
COPY data/ data/

RUN mkdir -p s2_organisation_and_version_control/s2/models

WORKDIR /
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync

ENTRYPOINT ["uv", "run", "s2_organisation_and_version_control/s2/src/s2_exercises/train.py"]