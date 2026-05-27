ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

ARG EXTRAS=""
RUN if [ -n "$EXTRAS" ]; then \
        pip install --no-cache-dir ".[${EXTRAS}]"; \
    fi

VOLUME ["/data"]
ENTRYPOINT ["lift"]
CMD ["--help"]