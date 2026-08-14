# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml README.md setup.py requirements.txt ./
COPY spin_rag ./spin_rag
COPY demo.py demo-data.txt ./
COPY assets ./assets
RUN pip install --no-cache-dir -e ".[app]"

FROM base AS test
COPY tests ./tests
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["pytest"]

FROM base AS runtime
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/')"
ENV HOST=0.0.0.0 PORT=8050 DASH_DEBUG=0
CMD ["python", "demo.py"]
