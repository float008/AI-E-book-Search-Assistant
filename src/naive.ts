import { Milvus } from "@langchain/community/vectorstores/milvus";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { IndexType } from "@zilliz/milvus2-sdk-node";
import "dotenv/config";
import z from "zod";

const COLLECTION_NAME = "ebook_collection";
const VECTOR_DIM = 1024;
const TOP_K = 5;

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
    question: state.question,
    k: state.k,
    strategy: route.strategy,
    routeReason: route.reason,
  };
};

const directAnswerNode = async (state: any) => {
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
    question: state.question,
    k: state.k,
    strategy: state.strategy,
    routeReason: state.routeReason,
    documents: [],
    generation,
  };
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

const retrieveNode = async (state: any) => {
  console.log("\n---检索相关内容---");
  const documents = await retrieveRelevantContent(state.question, state.k);
  if (documents.length === 0) {
    console.log("检索失败: 未找到相关文档");
  } else {
    console.log(`检索成功: 找到 ${documents.length} 个相关文档`);
    documents.forEach((item: any) => {
      console.log(`\n[片段 ${item.index}] 相似度: ${item.score.toFixed(4)}`);
      console.log(`书籍: ${item.book_id}`);
      console.log(`章节: 第 ${item.chapter_num} 章`);
      console.log(`片段索引: ${item.index}`);
      console.log(
        `内容: ${item.content.substring(0, 200)}${item.content.length > 200 ? "..." : ""}`,
      );
    });

    return {
      question: state.question,
      k: state.k,
      strategy: state.strategy,
      routeReason: state.routeReason,
      documents,
    };
  }
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

function decideNext(state: any) {
  return state.strategy === "simple" ? "direct_answer" : "retrieve";
}

const graph = new StateGraph(GraphState)
  .addNode("route_question", routeQuestionNode)
  .addNode("direct_answer", directAnswerNode)
  .addNode("retrieve", retrieveNode)
  .addNode("rag_generate", ragGenerateNode)
  .addEdge(START, "route_question")
  .addConditionalEdges("route_question", decideNext, {
    direct_answer: "direct_answer",
    retrieve: "retrieve",
  })
  .addEdge("retrieve", "rag_generate")
  .addEdge("direct_answer", END)
  .addEdge("rag_generate", END)
  .compile();

async function main() {
  //   const question = "雁门关事件的主谋，他的儿子最终结局是什么？";
  const question = "乔峰是男是女？";
  const kArg = 5;

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
    k: Number.isFinite(kArg) ? kArg : 5,
    strategy: "",
    routeReason: "",
    documents: [],
    generation: "",
  });

  console.log("\n【检索相关内容】");

  if (result.documents.length === 0) {
    console.log("未找到相关内容");
    console.log("\n【AI 回答】");
    console.log("抱歉，我没有找到相关的《天龙八部》内容。");
    return;
  } else {
    result.documents.forEach((item: any, i: number) => {
      console.log(`\n[片段 ${i + 1}] 相似度: ${item.score.toFixed(4)}`);
      console.log(`书籍: ${item.book_id}`);
      console.log(`章节: 第 ${item.chapter_num} 章`);
      console.log(`片段索引: ${item.index}`);
      console.log(
        `内容: ${item.content.substring(0, 200)}${item.content.length > 200 ? "..." : ""}`,
      );
    });
  }

  console.log(`\n最终策略: ${result.strategy}`);
  console.log(`\n最终原因: ${result.routeReason}`);

  if (!result.generation) {
    console.log("\n【AI 回答】");
    console.log("模型未返回内容。");
  }
}

main();
