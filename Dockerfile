# Light image: recorded mode + live CPU fallback, no torch.
# For the UNet on CPU, build with --build-arg WITH_ML=1 (installs requirements-ml.txt).
FROM python:3.12-slim

ARG WITH_ML=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt && \
    if [ "$WITH_ML" = "1" ]; then pip install -r requirements-ml.txt; fi

COPY holdpoint ./holdpoint
COPY app ./app
COPY config ./config
COPY examples ./examples
COPY tools ./tools

RUN useradd --create-home holdpoint && mkdir -p /app/results/tmp && chown -R holdpoint /app/results
USER holdpoint

EXPOSE 8501
CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
