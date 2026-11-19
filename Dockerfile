# Reproducible check image: build once (needs network), then run fully offline:
#   docker build -t audit-env .
#   docker run --rm --network none --cpus 2 --memory 4g audit-env make check
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

CMD ["make", "check"]
