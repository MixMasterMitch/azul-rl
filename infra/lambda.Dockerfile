FROM public.ecr.aws/lambda/python:3.11

# Install PyTorch CPU-only (smaller image)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir flask boto3 numpy pyyaml

# Copy application code
COPY agent/ agent/
COPY play/ play/

# Lambda handler
CMD ["play.lambda_handler.handler"]
