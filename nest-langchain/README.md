# nest-langchain

NestJS + LangChain Agent 后端，支持工具调用（用户查询、数据库 CRUD、发邮件、联网搜索、定时任务等）与多种对话接口。

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 框架 | NestJS |
| Agent | LangChain、LangGraph |
| 大模型 | OpenAI 兼容 API（如通义千问） |
| 对话协议 | Vercel AI SDK |
| 数据库 | MySQL + TypeORM |
| 定时任务 | `@nestjs/schedule` |
| 邮件 | Nodemailer |
| 联网搜索 | Bocha Web Search |

## 环境变量

在 `nest-langchain/` 下创建 `.env`：

```env
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your-api-key
OPENAI_STRUCTURED_MODEL=qwen-plus

# 可选
BOCHA_API_KEY=your-bocha-api-key
MAIL_HOST=smtp.example.com
MAIL_PORT=465
MAIL_SECURE=true
MAIL_USER=your-email@example.com
MAIL_PASS=your-password
MAIL_FROM=your-email@example.com
```

还需本地 MySQL（默认 `root/admin`，库名 `hello`，见 `src/app.module.ts`）。

## 启动

```bash
npm install
npm run start:dev
```

默认 `http://localhost:3000`。

## 主要接口

- `GET /ai/chat?query=` — 非流式问答
- `GET /ai/chat/stream?query=` — SSE 流式
- `POST /ai/chat` — 多轮 UI 对话（供 `frontend/` 使用）
- `CRUD /users` — 用户管理

SSE 测试页：`/sse-test.html`
