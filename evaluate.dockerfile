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

ENTRYPOINT ["uv", "run", "s2_organisation_and_version_control/s2/src/s2_exercises/evaluate.py"]

# docker run --name evaluate --rm -v /Users/carlahugod/Desktop/UNI/7sem/MLOps/dtu_mlops/s2_organisation_and_version_control/s2/models/model.pth:/s2_organisation_and_version_control/s2/models/model.pth -v /Users/carlahugod/Desktop/UNI/7sem/MLOps/dtu_mlops/data/processed/test_images.pt:/test_images.pt -v /Users/carlahugod/Desktop/UNI/7sem/MLOps/dtu_mlops/data/processed/test_targets.pt:/test_targets.pt evaluate:latest ../../models/model.pth \ 