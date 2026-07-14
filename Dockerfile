# Thin "extras" layer on top of a published base image. This is the
# Dockerfile most people build — it's fast because it just adds one pip
# layer on top of a base that's already built/pulled and cached.
#
# Usage:
#   docker build -t lift-cpu .                                  # base only, no extras
#   docker build --build-arg EXTRAS="cp-v3" -t lift-cpu .        # CPU segmentation
#   docker build --build-arg BASE=<dockerhub-user>/lift:gpu-base \
#                --build-arg EXTRAS="cp-sam,trackastra" -t lift-gpu .
#
# Any combination of extras works here — this is not limited to a fixed
# set of presets. See pyproject.toml [project.optional-dependencies] for
# the full list (cp-sam, cp-v3, trackastra, spotiflow, notebook, app).

ARG BASE=<dockerhub-user>/lift:cpu-base
FROM ${BASE}

ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then \
        pip install --no-cache-dir "live-foci[${EXTRAS}]"; \
    fi