# Volatile job layer on top of an already-loaded GPU base. Build context is
# the src, configs, and experiments trees.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY src /app/src
COPY configs /app/configs
COPY experiments /app/experiments
ENV PYTHONPATH=/app/src

