# COPV Studio Pro — compute backend container.
#
# The backend (JAX shell FEA + gmsh/OpenCASCADE meshing) is ~1.4 GB of scientific
# dependencies with 20-120 s solves and an in-memory mesh/JIT cache. That is a
# long-running server, not a serverless function — deploy this image on Render,
# Railway, Fly.io, Hugging Face Spaces, or any VM/container host, and point the
# Vercel-hosted static frontend at it with ?api=<backend-url>.
#
#   docker build -t copv-studio .
#   docker run -p 8081:8081 copv-studio
#
# See docs/DEPLOYMENT.md for the full split-deployment guide.

FROM python:3.11-slim

# gmsh's manylinux wheel links against X11/GL shared libraries even when headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/copv

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir .

ENV COPV_HOST=0.0.0.0 \
    COPV_PORT=8081 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

EXPOSE 8081
CMD ["python", "-m", "app.server"]
