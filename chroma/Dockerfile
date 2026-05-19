FROM chromadb/chroma:latest

# Ensure apt-get update completes without issues
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

