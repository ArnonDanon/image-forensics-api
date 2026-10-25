FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 IMGF_DATA_DIR=/data/images
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api,mcp]"

VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# REST API by default; for MCP over HTTP run:  python -m imgforensics.mcp_server --http --host 0.0.0.0
CMD ["uvicorn", "imgforensics.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
