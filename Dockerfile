# GraphForge 容器镜像（作者: 晨星）
FROM python:3.11-slim

LABEL org.opencontainers.image.title="GraphForge" \
      org.opencontainers.image.description="node2vec 图表示学习与链接预测系统" \
      org.opencontainers.image.authors="晨星" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY graphforge ./graphforge
COPY examples ./examples
COPY tests ./tests
COPY pyproject.toml README.md LICENSE ./

# 非 root 运行
RUN useradd -m -u 10001 graphforge && chown -R graphforge:graphforge /app
USER graphforge

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "graphforge.cli"]
CMD ["info"]
