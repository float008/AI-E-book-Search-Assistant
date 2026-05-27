import { ChatOpenAI } from "@langchain/openai";
import {
  AIMessage,
  createAgent,
  createMiddleware,
  HumanMessage,
} from "langchain";
import z from "zod";
import "dotenv/config";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createFilesystemMiddleware, FilesystemBackend } from "deepagents";

const permissions: any = [
  {
    operations: ["read"],
    paths: ["/secrets"],
    mode: "deny",
  },
  {
    operations: ["write"],
    paths: ["/todo.md"],
    mode: "allow",
  },
  {
    operations: ["write"],
    paths: ["/**"],
    mode: "deny",
  },
];

const workspaceDir = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "workspace",
);

fs.rmSync(workspaceDir, { recursive: true, force: true });
fs.mkdirSync(workspaceDir);
fs.writeFileSync(path.join(workspaceDir, "secret.txt"), "机密：11", "utf8");

const loggingMiddleware = createMiddleware({
  name: "LoggingMiddleware",
  stateSchema: z.object({
    modelCallCount: z.number().default(0),
  }),
  beforeAgent: (state) => {
    console.log("\n[LoggingMiddleware] beforeAgent", state.messages.length);
  },
  beforeModel: (state) => {
    console.log("\n[LoggingMiddleware] beforeModel", state.messages.length);
  },
  afterModel: (state) => {
    console.log("\n[LoggingMiddleware] afterModel", state.messages.length);
  },
  afterAgent: (state) => {
    console.log("\n[LoggingMiddleware] afterAgent", state.messages.length);
  },
});

const addContextMiddleware = createMiddleware({
  name: "AddContextMiddleware",
  wrapModelCall: async (request, handler) => {
    console.log("[AddContextMiddleware] 注入额外上下文");
    return handler({
      ...request,
      systemMessage: request.systemMessage.concat("\n\n请用一句话简洁回答。"),
    });
  },
});

const blockedContentMiddleware = createMiddleware({
  name: "BlockedContentMiddleware",
  beforeModel: {
    canJumpTo: ["end"],
    hook: (state) => {
      const last = state.messages.at(-1);
      const text =
        typeof last?.content === "string"
          ? last.content
          : String(last?.content ?? "");

      if (text.includes("BLOCKED")) {
        console.log("[BlockedContentMiddleware] 内容被屏蔽");
        return {
          messages: [new AIMessage("该请求已被 middleware 拦截。")],
          jumpTo: "end",
        };
      }
    },
  },
});

const model = new ChatOpenAI({
  model: process.env.OPENAI_BASE_MODEL,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  temperature: 0,
});

const agent = createAgent({
  model,
  tools: [],
  systemPrompt:
    "工作区根路径为 /。用 ls、read_file、write_file、edit_file 操作文件，路径以 / 开头。中文回答。",
  middleware: [
    createFilesystemMiddleware({
      backend: new FilesystemBackend({
        rootDir: workspaceDir,
        virtualMode: true,
      }),
      permissions,
    }),
  ],
});

// const result = await agent.invoke({
//   messages: [new HumanMessage("今天天气怎么样？")],
// });

const run = async (label: string, prompt: string) => {
  console.log(`\n=== ${label} ===\n`, prompt, "\n");
  const { messages }: { messages: any } = await agent.invoke(
    { messages: [new HumanMessage(prompt)] },
    { recursionLimit: 20 },
  );
  for (const m of messages) {
    for (const t of m.tool_calls ?? []) console.log("→", t.name);
  }
  console.log("回复:", messages.at(-1)?.content);
};

async function expectDenied(label: string, prompt: string) {
  console.log(`\n=== ${label}（预期拒绝）===\n`, prompt, "\n");
  try {
    await agent.invoke(
      { messages: [new HumanMessage(prompt)] },
      { recursionLimit: 5 },
    );
    console.log("未触发拒绝（异常）");
  } catch (e: any) {
    const msg = e.cause?.message ?? e.message;
    console.log("✗", msg);
  }
}

await run(
  "允许的操作",
  "write_file 创建 /todo.md（三条待办），edit_file 把第一条标为完成，ls /，一句话总结。",
);

await expectDenied("禁止读", "只调用 read_file，路径 /secret.txt。");
await expectDenied("禁止写", "只调用 write_file，路径 /hack.txt，内容 test。");
