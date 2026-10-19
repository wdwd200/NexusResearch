# NexusResearch 后端镜像
# 单阶段构建：依赖从 uv.lock 精确还原，运行时不带开发工具

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# 运行期依赖：curl 用于健康检查，其余为 Python 包编译/运行所需的最小集合
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 uv，再用 lock 文件还原依赖，充分利用镜像层缓存
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 复制业务代码与知识库文档
COPY app ./app
COPY scripts ./scripts
COPY docs/knowledge_base ./docs/knowledge_base

# 运行期目录：状态账本、会话产物、上传暂存
RUN mkdir -p /app/data /app/output /app/updated

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8002/api/health || exit 1

# 单 worker 异步模式：4C4G 机器上避免多进程各自持有状态账本连接
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
