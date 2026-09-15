ARG LAMBDA_PYTHON=3.12
FROM public.ecr.aws/lambda/python:${LAMBDA_PYTHON} AS astra-builder

# Compile against the same OS as the runtime, using portable CPU instructions.
RUN if command -v dnf >/dev/null 2>&1; then \
      dnf install -y gcc gcc-c++ make && dnf clean all; \
    else \
      yum install -y gcc gcc-c++ make && yum clean all; \
    fi
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain 1.90.0
ENV PATH="/root/.cargo/bin:${PATH}" RUSTFLAGS="-C target-cpu=x86-64"
RUN pip install --no-cache-dir maturin==1.9.6
WORKDIR /build/astra
COPY native/astra/Cargo.toml native/astra/Cargo.lock native/astra/pyproject.toml native/astra/rust-toolchain.toml ./
COPY native/astra/src/ ./src/
RUN maturin build --release --locked --out /wheels

FROM public.ecr.aws/lambda/python:${LAMBDA_PYTHON} AS runtime

# Install PyTorch CPU-only (smaller image)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
COPY pyproject.toml .
COPY infra/requirements-runtime.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements-runtime.txt

# Keep heavyweight Python dependencies cached when the native implementation changes.
COPY --from=astra-builder /wheels/ /tmp/astra-wheels/
RUN pip install --no-cache-dir /tmp/astra-wheels/*.whl && rm -rf /tmp/astra-wheels

# Copy application code
COPY agent/ agent/
COPY play/ play/

ENV AZUL_MODEL_REGISTRY=/var/task/play/artifacts/registry.json \
    AZUL_REQUIRE_MODEL=1 AZUL_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m play.scripts.prepare_release --validate-only --output /var/task/play/artifacts

# Lambda handler
CMD ["play.lambda_handler.handler"]
