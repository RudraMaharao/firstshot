# One image for local Docker and AWS Lambda.
# On Lambda, the Lambda Web Adapter extension forwards events to uvicorn on port 8080,
# so the same FastAPI app runs unchanged.
FROM public.ecr.aws/docker/library/python:3.12-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

# MALLOC_* keeps freed NumPy buffers in the heap instead of handing them back to the OS.
# That avoids page-fault storms on small VMs; Lambda runs on Firecracker microVMs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/api/health \
    MALLOC_MMAP_THRESHOLD_=134217728 \
    MALLOC_TRIM_THRESHOLD_=536870912 \
    MALLOC_TOP_PAD_=268435456

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY firstshot/ firstshot/
COPY service/ service/
COPY results/benchmark.json results/benchmark.json
RUN python -m compileall -q firstshot service

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8080"]
