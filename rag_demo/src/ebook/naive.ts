import { Milvus } from "@langchain/community/vectorstores/milvus";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { IndexType } from "@zilliz/milvus2-sdk-node";
import "dotenv/config";
import z from "zod";

const COLLECTION_NAME = "ebook_collection";
const VECTOR_DIM = 1024;
const TOP_K = 8;

const RouteSchema = z.object({
  strategy: z.enum(["simple", "complex"]),
  reason: z.string(),
});

const GraphState = Annotation.Root({
  question: Annotation,
  k: Annotation,
  strategy: Annotation,
  routeReason: Annotation,
  documents: Annotation,
  generation: Annotation,
  /** 拆解得到的有序子问题，仅用于检索 */
  subQuestions: Annotation,
  /** 下一轮 retrieve 要用的下标（指向 subQuestions 中尚未检索的那一条） */
  nextSubIdx: Annotation,
  /** 当前检索的子问题 */
  currentQuery: Annotation,
  /** 已经完成的检索轮数（从 0 开始，每进一次 retrieve 加 1）。 */
  retrievalCount: Annotation,
  /** 最大检索轮数（防止无限循环） */
  maxRetrievals: Annotation,
  /** 规划节点决定的下一步走向："retrieve" 或 "rag_generate"。 */
  plannedNext: Annotation,
});

const model = new ChatOpenAI({
  temperature: 0,
  model: process.env.OPENAI_STRUCTURED_MODEL,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.OPENAI_EMBEDDINGS_MODEL,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  dimensions: VECTOR_DIM,
});

let vectorStore: any;
const DecomposeSchema = z.object({
  sub_questions: z.array(z.string()).min(1).max(8),
  reason: z.string(),
});

const NextStepSchema = z.object({
  nextAction: z.enum(["generate", "retrieve"]),
  reason: z.string(),
});

const mergeUnique = (existingDocs: any[], newDocs: any[]) => {
  const map = new Map();
  for (const d of [...existingDocs, ...newDocs]) {
    const key = String(d.id);
    const prev = map.get(key);
    if (!prev || Number(d.score) > Number(prev.score)) {
      map.set(key, d);
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => Number(b.score) - Number(a.score),
  );
};
const retrieveRelevantContent = async (question: string, k = TOP_K) => {
  try {
    const docsWithScores = await vectorStore.similaritySearchWithScore(
      question,
      k,
    );

    return docsWithScores.map(([doc, score]: [any, number]) => ({
      score,
      content: doc.pageContent,
      id: doc.metadata?.id ?? "unknown",
      book_id: doc.metadata?.book_id ?? "未知",
      chapter_num: doc.metadata?.chapter_num ?? "未知",
      index: doc.metadata?.index ?? "未知",
    }));
  } catch (error: any) {
    console.error("检索内容时出错:", error.message);
    return [];
  }
};
const routeQuestionNode = async (state: any) => {
  console.log("---问答路由器---");
  const router = model.withStructuredOutput(RouteSchema);
  const route = await router.invoke(`
  你是问答路由器。请判断用户问题是否需要外部检索。
  
  规则：
  - simple: 常识问答、简短定义、无需特定小说细节即可回答。
  - complex: 需要《天龙八部》具体情节、人物关系、章节事实、原文细节或证据支持。
  
  用户问题：${state.question}
  `);

  console.log("route路由器", route);

  return {
    strategy: route.strategy,
    routeReason: route.reason,
    retrievalCount: 0,
    maxRetrievals: state.maxRetrievals ?? TOP_K,
    documents: [],
    subQuestions: [],
    nextSubIdx: 0,
    currentQuery: "",
  };
};

const simpleAnswerNode = async (state: any) => {
  console.log("---简单问题直接回答---");
  process.stdout.write("\n【AI 回答（流式）】\n");
  let generation = "";
  const stream = await model.stream(`你是一个中文问答助手，请直接简洁回答问题。
  
  问题：${state.question}
  `);
  for await (const chunk of stream) {
    const text = typeof chunk.content === "string" ? chunk.content : "";
    if (!text) continue;
    generation += text;
    process.stdout.write(text);
  }
  process.stdout.write("\n");
  return {
    generation,
  };
};

const decomposeQuestionNode = async (state: any) => {
  console.log("\n---拆解问题---");

  const decomposer = model.withStructuredOutput(DecomposeSchema);
  const out =
    await decomposer.invoke(`你是《天龙八部》多跳问答的「子问题拆解器」。

用户原始问题：
${state.question}

任务：将问题拆成**有序**子问题列表 sub_questions，用于**依次向量检索**。要求：
1. 链式推理、多层关系、因果先后的问题，必须拆成多条；单跳即可答的也可只输出 1 条。
2. 每条子问题必须是**可独立检索**的完整中文问句，**禁止**使用「他/她/此人/上文」等指代；可写全人物名与事件名。
3. 顺序必须符合推理链：先搞清前置实体/事实，再查后续结论。
4. **不要**把整句原题原样复制成唯一一条（除非确实无法拆分）；不要拆成过碎的关键词列表。
5. 输出 1～8 条即可。

请输出 sub_questions 与简短 reason。`);

  const subQuestions = out.sub_questions.map((s) => s.trim()).filter(Boolean);
  if (subQuestions.length === 0) {
    throw new Error("decompose_question: sub_questions 为空");
  }

  console.log(`拆解 ${subQuestions.length} 条子问题 (${out.reason})`);
  subQuestions.forEach((q, i) => {
    console.log(`  [${i + 1}] ${q}`);
  });

  return {
    subQuestions,
    nextSubIdx: 0,
    currentQuery: subQuestions[0],
  };
};

const retrieveNode = async (state: any) => {
  console.log("\n---检索当前子问题相关内容---");

  const subs = state.subQuestions ?? [];
  const idx = state.nextSubIdx ?? 0;
  const q = subs[idx]?.trim();
  if (!q) {
    throw new Error(
      `retrieve: 子问题下标 ${idx} 无有效文本（共 ${subs.length} 条）`,
    );
  }

  const round = state.retrievalCount + 1;
  console.log(
    `---RETRIEVE (第 ${round} 轮，子问题 ${idx + 1}/${subs.length})---`,
  );
  console.log(`查询: ${q}`);

  const newDocs = await retrieveRelevantContent(q, state.k);
  const merged = mergeUnique(state.documents ?? [], newDocs);

  if (newDocs.length === 0) {
    console.log("检索失败: 未找到相关文档");
  } else {
    console.log(
      `本轮命中 ${newDocs.length} 条，累计去重后 ${merged.length} 条`,
    );
    newDocs.forEach((item: any, i: number) => {
      const preview =
        item.content.length > 120
          ? `${item.content.substring(0, 120)}...`
          : item.content;
      console.log(
        `[R${i + 1}] score=${Number(item.score).toFixed(4)} chapter=${item.chapter_num} index=${item.index}`,
      );
      console.log(`      ${preview}`);
    });

    return {
      documents: merged,
      retrievalCount: round,
      nextSubIdx: idx + 1,
      currentQuery: q,
    };
  }
};

const planNextStepNode = async (state: any) => {
  console.log("\n---规划下一步---");
  const subs = state.subQuestions ?? [];
  const nextIdx = state.nextSubIdx ?? 0;
  const remaining = subs.length - nextIdx;

  const subList = subs
    .map(
      (s: string, i: number) =>
        `${i + 1}. ${s}${i < nextIdx ? " （已检索）" : i === nextIdx ? " （下一轮将检索，若选择继续）" : " （未检索）"}`,
    )
    .join("\n");

  const docStr =
    state.documents.length === 0
      ? "（尚无检索结果）"
      : state.documents
          .slice(0, 6)
          .map(
            (d: any, i: number) =>
              `[${i + 1}] score=${Number(d.score).toFixed(4)} 第${d.chapter_num}章: ${d.content.slice(0, 200)}${d.content.length > 200 ? "..." : ""}`,
          )
          .join("\n\n");

  const prompt = `你是多跳 RAG 规划器。检索查询已由前置步骤拆解为**有序子问题**；若需继续检索，下一轮将自动使用「下一条子问题」做向量检索，你**不要**自拟新的检索句。

              用户原始问题：${state.question}
              
              子问题序列：
              ${subList || "（无）"}
              
              已检索轮数：${state.retrievalCount}；剩余未检索子问题条数：${remaining}
              最大检索轮数上限：${state.maxRetrievals}
              
              已召回文档摘要：
              ${docStr}
              
              请判断下一步：
              1) 已有足够依据回答用户原始问题 → nextAction=generate
              2) 仍缺关键事实、且仍存在未检索的子问题、且未超过轮数上限 → nextAction=retrieve
              
              硬性规则：
              - 若剩余未检索子问题条数为 0，必须 nextAction=generate。
              - 若已检索轮数已达到或超过最大检索轮数，必须 nextAction=generate。`;

  const planner = model.withStructuredOutput(NextStepSchema);
  const { nextAction, reason } = await planner.invoke(prompt);

  let finalNext = nextAction;
  if (state.retrievalCount >= state.maxRetrievals) finalNext = "generate";
  if (remaining <= 0) finalNext = "generate";

  console.log(
    `[决策] plannedNext=${finalNext} (模型建议=${nextAction}) (${reason})`,
  );

  return {
    plannedNext: finalNext,
  };
};

const ragGenerateNode = async (state: any) => {
  console.log("\n---RAG生成回答---");
  const context = state.documents
    .map(
      (item: any, i: number) =>
        `[片段 ${i + 1}]
章节: 第 ${item.chapter_num} 章
内容: ${item.content}`,
    )
    .join("\n\n━━━━━\n\n");

  const prompt = `你是一个专业的《天龙八部》小说助手。基于小说内容回答问题，用准确、详细的语言。

请根据以下《天龙八部》小说片段内容回答问题：
${context || "没有找到相关内容"}

用户问题: ${state.question}

回答要求：
1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答
2. 可以综合多个片段的内容，提供完整的答案
3. 如果片段中没有相关信息，请如实告知用户
4. 回答要准确，符合小说的情节和人物设定
5. 可以引用原文内容来支持你的回答

AI 助手的回答:`;

  process.stdout.write("\n【AI 回答（流式）】\n");
  let generation = "";
  const stream = await model.stream(prompt);
  for await (const chunk of stream) {
    const text = typeof chunk.content === "string" ? chunk.content : "";
    if (!text) continue;
    generation += text;
    process.stdout.write(text);
  }
  process.stdout.write("\n");

  return {
    question: state.question,
    k: state.k,
    strategy: state.strategy,
    routeReason: state.routeReason,
    documents: state.documents,
    generation,
  };
};

const afterRoute = (state: any) => {
  return state.strategy === "simple" ? "simple_answer" : "decompose_question";
};

/** 返回值必须是 addConditionalEdges 映射里的键名，与节点 rag_generate 对应 */
const afterPlan = (state: any) => {
  return state.plannedNext === "retrieve" ? "retrieve" : "rag_generate";
};

const graph = new StateGraph(GraphState)
  .addNode("route_question", routeQuestionNode)
  .addNode("simple_answer", simpleAnswerNode)
  .addNode("decompose_question", decomposeQuestionNode)
  .addNode("retrieve", retrieveNode)
  .addNode("plan_next_step", planNextStepNode)
  .addNode("rag_generate", ragGenerateNode)
  .addEdge(START, "route_question")
  .addConditionalEdges("route_question", afterRoute, {
    simple_answer: "simple_answer",
    decompose_question: "decompose_question",
  })
  .addEdge("decompose_question", "retrieve")
  .addEdge("retrieve", "plan_next_step")
  .addConditionalEdges("plan_next_step", afterPlan, {
    retrieve: "retrieve",
    rag_generate: "rag_generate",
  })
  .addEdge("simple_answer", END)
  .addEdge("rag_generate", END)
  .compile();

async function main() {
  const question =
    "《天龙八部》中「四大恶人」排行第二的是谁？此人之子在身世揭晓前，其生父在武林中的公开身份是什么？";

  const drawable = await graph.getGraphAsync();
  const mermaid = drawable.drawMermaid({ withStyles: true });
  console.log(mermaid);

  console.log("连接到 Milvus...");
  vectorStore = await Milvus.fromExistingCollection(embeddings, {
    collectionName: COLLECTION_NAME,
    url: "localhost:19530",
    textField: "content",
    primaryField: "id",
    vectorField: "vector",
    indexCreateOptions: {
      metric_type: "COSINE",
      index_type: IndexType.IVF_FLAT,
    },
  });
  vectorStore.indexSearchParams = {
    metric_type: "COSINE",
  };
  console.log("✓ 已连接\n");

  try {
    await vectorStore.client.loadCollection({
      collection_name: COLLECTION_NAME,
    });
    console.log(`✓ 集合 ${COLLECTION_NAME} 已加载\n`);
  } catch (error: any) {
    if (!error.message.includes("already loaded")) {
      throw error;
    }
    console.log(`✓ 集合 ${COLLECTION_NAME} 已处于加载状态\n`);
  }

  console.log("=".repeat(80));
  console.log(`问题: ${question}`);
  console.log("=".repeat(80));

  const result: any = await graph.invoke({
    question,
    k: 5,
    strategy: "",
    routeReason: "",
    subQuestions: [],
    nextSubIdx: 0,
    documents: [],
    currentQuery: "",
    retrievalCount: 0,
    maxRetrievals: TOP_K,
    plannedNext: "",
    generation: "",
  });

  console.log("\n【检索相关内容】");

  if (result.strategy === "complex") {
    if (result.subQuestions?.length) {
      console.log("\n【子问题序列】");
      result.subQuestions.forEach((s: string, i: number) =>
        console.log(`  ${i + 1}. ${s}`),
      );
    }

    if (result.documents.length === 0) {
      console.log("未找到相关内容");
    } else {
      result.documents.forEach((item: any, i: number) => {
        console.log(
          `\n[片段 ${i + 1}] 相似度: ${Number(item.score).toFixed(4)}`,
        );
        console.log(`书籍: ${item.book_id}`);
        console.log(`章节: 第 ${item.chapter_num} 章`);
        console.log(`片段索引: ${item.index}`);
        console.log(
          `内容: ${item.content.substring(0, 200)}${item.content.length > 200 ? "..." : ""}`,
        );
      });
    }
    console.log(
      `\n检索轮数: ${result.retrievalCount} / ${result.maxRetrievals}`,
    );
  }

  console.log(`\n最终策略: ${result.strategy}`);
  console.log(`\n最终原因: ${result.routeReason}`);

  if (!result.generation?.trim()) {
    console.log("\n【AI 回答】");
    console.log("模型未返回内容。");
  }
}

main();
