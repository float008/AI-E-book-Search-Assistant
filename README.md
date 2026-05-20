# AI-E-book-Search-Assistant

基于大模型与多存储的 **RAG / Agent / 图谱 / 评测** 实验项目。涵盖 EPUB 电子书问答、Elasticsearch + Milvus 混合检索、Neo4j 图谱问答、LangSmith 自动化评测等。

## 技术栈

| 类别      | 技术                          |
| --------- | ----------------------------- |
| 语言      | TypeScript、Node.js           |
| 编排      | LangGraph、LangChain          |
| 大模型    | OpenAI 兼容 API（如通义千问） |
| Embedding | text-embedding-v3（1024 维）  |
| 向量库    | Milvus                        |
| 全文检索  | Elasticsearch 8 + IK 中文分词 |
| 图数据库  | Neo4j                         |
| 评测      | LangSmith、OpenEvals          |
| 重排      | 通义 DashScope Rerank         |
| 配置      | dotenv                        |

## 能力模块

| 模块       | 说明                                            |
| ---------- | ----------------------------------------------- |
| 电子书 RAG | EPUB 分块入库，向量检索与基础问答               |
| 多跳 Agent | 路由、子问题拆解、多轮检索与规划生成            |
| 混合检索   | 查询改写，全文 + 向量并行召回，重排后作答       |
| 客服知识库 | Markdown/TXT 入库 Milvus，检索增强问答          |
| 图谱 RAG   | 自然语言转 Cypher，查图后生成答案               |
| RAG 评测   | 数据集回归 + 忠实度 / 有用性 / 检索相关性等指标 |

## 基础设施

```bash
docker compose up -d
```

| 服务          | 端口        | 说明                     |
| ------------- | ----------- | ------------------------ |
| Elasticsearch | 9200        | 全文检索（内置 IK 分词） |
| Kibana        | 5601        | ES 控制台                |
| Milvus        | 19530       | 向量检索                 |
| MinIO         | 9000 / 9001 | 对象存储                 |
| Neo4j         | 7474 / 7687 | 图库 Browser / Bolt      |

Neo4j 默认账号：`neo4j` / `12345678`。数据持久化在项目 `volumes` 目录。

首次构建 ES 镜像若失败，可执行 `docker compose build --no-cache es` 后重试。

## 环境变量

在项目根目录创建 `.env`（勿提交密钥）：

```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your-api-key
OPENAI_BASE_MODEL=qwen-plus
OPENAI_STRUCTURED_MODEL=qwen-plus
OPENAI_EMBEDDINGS_MODEL=text-embedding-v3

OPENAI_RERANK_MODEL=qwen3-rerank
OPENAI_RERANK_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank

LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=rag_demo
LANGCHAIN_TRACING_V2=true

MILVUS_COLLECTION=rag_docs
MILVUS_URI=http://localhost:19530
```

结构化任务建议使用支持 function calling 的模型；批量评测时较小模型通常更快。

## 快速开始

```bash
npm install
docker compose up -d
```

在**项目根目录**运行各模块入口脚本（使用 `tsx`）。客服知识库原文放在根目录 `data/` 下（Markdown/TXT）。

| 场景     | 大致步骤                                               |
| -------- | ------------------------------------------------------ |
| 电子书   | 准备 EPUB → 入库 → 检索 / 问答 / 多跳 Agent            |
| 混合检索 | 样例数据写入 ES 与 Milvus → 运行混合检索流水线         |
| 客服 RAG | 文档入库 → 命令行问答                                  |
| 评测     | 上传评测集（首次）→ 批量跑评测 → 在 LangSmith 查看报告 |
| 图谱     | 启动 Neo4j → 图数据示例或 GraphRAG 问答                |

```bash
npm run build
```

## 架构示意

### 电子书多跳 Agent

```
EPUB → 入库分块 → Milvus
用户问题 → Agent 工作流 → Milvus + 大模型 → 回答
```

### 混合检索

```
用户问题 → 查询改写 ─┬→ 全文检索 ─┐
                    └→ 向量检索 ─┴→ 合并去重 → 重排 → 生成答案
```

### RAG 评测

```
评测数据集 → 被测 RAG → Milvus
                ↓
         答案 + 检索上下文 → LLM 裁判打分 → LangSmith 报告
```

评测指标含义：

| 指标       | 含义                     |
| ---------- | ------------------------ |
| 忠实度     | 答案是否有检索上下文支撑 |
| 有用性     | 是否切题、有用           |
| 检索相关性 | 召回片段与问题是否相关   |

## License

ISC
