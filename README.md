# NexusResearch

多智能体协同投研与自动化报告生成系统。面向企业内部研究场景，把「公网检索 + 业务数据查询 + 私有知识库检索」三路信息汇总为可下载的 Markdown / PDF 研报。

## 核心能力

- **一主三从协同**：主智能体负责任务拆解、子智能体调度与证据汇总，三个子智能体各自封装一类信息源，只回传提炼后的结构化摘要，避免原始检索数据撑爆主上下文。
- **内部知识库**：文档切片 → 硅基流动 `bge-large-zh-v1.5` 向量化 → Qdrant 存储 → 向量粗召回 Top-15 → `bge-reranker-v2-m3` 精排 Top-5，全链路不依赖外部重型 RAG 服务。
- **上下文治理**：稳定层（System 契约）、当前层（滑动窗口对话）、证据层（持久化账本）、临时层（重试草稿）四层分离；`budget_messages` 按字符预算逆序截取，`pair_tool_messages` 修复被截断的 `assistant → tool` 配对，避免大模型接口 400。
- **状态账本与断点恢复**：SQLite 持久化 Checkpointer 替换内存级 `InMemorySaver`，服务重启或断线后按 `thread_id` 恢复。
- **人工确认**：正式 PDF 生成前中断挂起，推送待审批动作，前端确认或填写修改意见后携带同一 `thread_id` 恢复执行。
- **异步任务与实时推送**：`asyncio.Task` 管理长耗时任务，`/ws/{thread_id}` 推送子智能体调度、工具耗时与最终结论；`/cancel` 支持主动取消。

## 技术栈

| 层次 | 技术 |
| --- | --- |
| 编排 | DeepAgents 0.5.7 / LangGraph 1.1.10 / LangChain 1.2.17 |
| 服务 | FastAPI + uvicorn（单 worker 异步）+ WebSocket |
| 模型 | DeepSeek OpenAI 兼容接口 |
| 检索 | Qdrant 1.16 + 硅基流动 Embedding / Rerank |
| 数据 | MySQL 8.0（业务库）+ SQLite（状态账本） |
| 交付 | ReportLab（中文字体与多级排版）+ FastAPI FileResponse |
| 前端 | React 19 + Vite 7 + Ant Design 5 |

## 目录结构

```text
nexus-research-agents/
├── app/
│   ├── agent/
│   │   ├── context_manager.py      # 四层上下文治理与滑动窗口中间件
│   │   ├── checkpointer.py         # SQLite 状态账本生命周期
│   │   ├── llm.py                  # OpenAI 兼容模型初始化
│   │   ├── main_agent.py           # 主智能体组装、任务启动与断点恢复
│   │   ├── prompts.py              # 读取 prompt/prompts.yml
│   │   └── subagents/              # 网络搜索 / 数据库 / 内部知识库三个子智能体
│   ├── api/
│   │   ├── context.py              # ContextVar 保存 thread_id 与 session_dir
│   │   ├── monitor.py              # 工具、子智能体、审批事件推送
│   │   └── server.py               # 任务、恢复、状态、上传下载、WebSocket 接口
│   ├── rag/
│   │   ├── embedding_client.py     # Embedding 与 Rerank 客户端
│   │   └── qdrant_store.py         # 知识库建表、写入、两阶段检索
│   ├── tools/                      # 知识库、MySQL、Tavily、文件读写、Markdown、PDF
│   ├── utils/                      # 路径解析与 Markdown→PDF 渲染
│   ├── config.py                   # 统一环境变量与配置快照
│   └── prompt/prompts.yml          # 主/子智能体提示词
├── scripts/ingest_knowledge_base.py  # 知识库切片入库脚本
├── docker/mysql/mysql.sql          # 业务库初始化数据
├── docs/knowledge_base/            # 内部知识库原始文档（按专题分目录）
├── deploy/nginx.conf               # 生产反向代理与静态托管配置
├── frontend/                       # React 前端
├── Dockerfile
├── docker-compose.prod.yaml        # 4C4G 生产编排
└── pyproject.toml / uv.lock
```

## 快速开始

### 1. 环境要求

- Python 3.12、[uv](https://github.com/astral-sh/uv)
- Node.js 20+、pnpm 10
- Docker（Qdrant 与 MySQL；本地已有服务时可跳过）
- DeepSeek API Key、硅基流动 API Key、Tavily API Key

### 2. 配置环境变量

```bash
cp .env.example .env
```

必填项：

| 变量 | 说明 |
| --- | --- |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_NAME` | 对话模型。可用名称以当前账号和供应商实际支持为准 |
| `SILICONFLOW_API_KEY` | Embedding 与 Rerank。模型名需通过 `app/config.py` 白名单，价格以供应商为准 |
| `TAVILY_API_KEY` | 联网搜索 |
| `MYSQL_*` | 业务库连接 |
| `QDRANT_URL` | 向量库地址，默认 `http://localhost:6333` |
| `ACCESS_TOKEN` | 可选演示口令。配置后 HTTP 走 Bearer，WebSocket 走握手参数。不是用户隔离 |

### 3. 启动依赖服务

```bash
# 仅启动 Qdrant
docker run -d --name nexus-qdrant -p 6333:6333 -v nexus_qdrant:/qdrant/storage qdrant/qdrant:v1.16

# 启动开发用 MySQL（含初始化数据）
docker compose -f docker/docker-compose.yaml up -d
```

### 4. 安装依赖并入库知识库

```bash
uv sync
uv run python scripts/ingest_knowledge_base.py
```

脚本会扫描 `docs/knowledge_base/<专题>/` 下的 PDF / Markdown / TXT，按 400 字切片、60 字重叠写入 Qdrant 的 `internal_kb` 集合。切片长度受 bge 模型 512 token 输入上限约束，不要随意调大。加 `--reset` 可全量重建。

### 5. 启动后端与前端

```bash
uv run uvicorn app.api.server:app --host 0.0.0.0 --port 8002 --reload

cd frontend
pnpm install
pnpm dev
```

前端默认访问 `http://localhost:5173`，Vite 会把 `/api` 与 `/ws` 代理到 `8002`。

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/task` | 启动任务，返回 `thread_id` |
| POST | `/api/task/{thread_id}/resume` | 人工确认后恢复：`{"decision": "approve"\|"reject", "feedback": "..."}` |
| GET | `/api/task/{thread_id}/status` | 查询运行中 / 待审批状态 |
| POST | `/api/task/{thread_id}/cancel` | 取消任务 |
| POST | `/api/upload` | 上传会话附件 |
| GET | `/api/files` / `/api/download` | 列出与下载产物 |
| GET | `/api/health` | 健康检查 |
| WS | `/ws/{thread_id}` | 实时事件流 |

事件类型：`session_created`、`tool_start`、`assistant_call`、`awaiting_approval`、`approval_resolved`、`task_result`、`task_cancelled`、`error`。

## 生产部署（4C4G3M）

```bash
# 1. 构建前端产物。生产 nginx 直接挂载 frontend/dist，不必再复制到 deploy/web
cd frontend && pnpm install && pnpm build && cd ..

# 2. 配置生产环境变量
cp .env.example .env   # 填入真实密钥

# 3. 启动
docker compose -f docker-compose.prod.yaml up -d --build
```

编排 limits 为 MySQL 600M、Qdrant 256M、API 512M、Nginx 64M，合计 1432M。这是容器上限，不是已测常驻内存。Nginx 开启 `gzip` 与 `gzip_static`，适配 3Mbps 带宽。

## 设计说明

### 为什么不使用外部 RAG 服务

外部 RAG 服务通常需要 16GB 以上内存，在 4C4G 机器上无法与其他服务共存。本项目的替代方案把检索拆成两个正交步骤：Qdrant 负责向量粗召回，硅基流动 Rerank 负责交叉注意力精排。效果上覆盖了「召回 + 重排」标准范式，额外内存开销接近零。

### 上下文为什么分四层

长程多轮调用中，真正需要进入当次请求的只有「系统契约 + 最近对话」。完整历史留在状态账本里按需恢复，校验重试的中间草稿只在局部流转。这样既控制了 Token 消耗，也避免把错误草稿写进长期记忆。

### 断点恢复为什么用 SQLite

单机部署下 SQLite 足够可靠，且不需要额外进程。它只承诺恢复已挂起的人工确认检查点，不会把中断中的模型调用自动重放成通用队列。注意必须使用 `AsyncSqliteSaver`：同步版 `SqliteSaver` 不支持 `astream`。WAL 模式让并发会话读写不互相阻塞。

## License

MIT License. 详见 [LICENSE](LICENSE)。
