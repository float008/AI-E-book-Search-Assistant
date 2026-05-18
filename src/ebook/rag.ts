import "dotenv/config";
import { MilvusClient, MetricType } from "@zilliz/milvus2-sdk-node";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

const COLLECTION_NAME = "ebook_collection";
const VECTOR_DIM = 1024;

const client = new MilvusClient({
  address: "localhost:19530",
});

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.OPENAI_EMBEDDINGS_MODEL,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  dimensions: VECTOR_DIM,
});

const model = new ChatOpenAI({
  temperature: 0.7,
  model: process.env.OPENAI_BASE_MODEL,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

async function getEmbedding(text: string): Promise<number[]> {
  return embeddings.embedQuery(text);
}

interface EbookSearchHit {
  id: string;
  book_id: string;
  chapter_num: number;
  index: number;
  content: string;
  score: number;
}

async function retrieveRelevantContent(
  question: string,
  k: number,
): Promise<EbookSearchHit[]> {
  try {
    const queryVector = await getEmbedding(question);

    const searchResult = await client.search({
      collection_name: COLLECTION_NAME,
      vector: queryVector,
      limit: k,
      metric_type: MetricType.COSINE,
      output_fields: ["id", "book_id", "chapter_num", "index", "content"],
    });

    return searchResult.results as EbookSearchHit[];
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("检索内容时出错:", message);
    return [];
  }
}

function messageContentToString(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((c) =>
        typeof c === "object" && c && "text" in c
          ? String((c as { text: unknown }).text)
          : String(c),
      )
      .join("");
  }
  return String(content);
}

async function answerEbookQuestion(
  question: string,
  k: number,
): Promise<string> {
  console.log("=".repeat(80));
  console.log(`问题: ${question}`);
  console.log("=".repeat(80));

  console.log("\n【检索相关内容】");
  const retrievedContent = await retrieveRelevantContent(question, k);

  if (retrievedContent.length === 0) {
    console.log("未找到相关内容");
    return "抱歉，我没有找到相关的《天龙八部》内容。";
  }

  retrievedContent.forEach((item, i) => {
    console.log(`\n[片段 ${i + 1}] 相似度: ${item.score.toFixed(4)}`);
    console.log(`书籍: ${item.book_id}`);
    console.log(`章节: 第 ${item.chapter_num} 章`);
    console.log(`片段索引: ${item.index}`);
    console.log(
      `内容: ${item.content.substring(0, 200)}${item.content.length > 200 ? "..." : ""}`,
    );
  });

  const context = retrievedContent
    .map((item, i) => {
      return `[片段 ${i + 1}]
章节: 第 ${item.chapter_num} 章
内容: ${item.content}`;
    })
    .join("\n\n━━━━━\n\n");

  const prompt = `你是一个专业的《天龙八部》小说助手。基于小说内容回答问题，用准确、详细的语言。

        请根据以下《天龙八部》小说片段内容回答问题：
        ${context}
        
        用户问题: ${question}
        
        回答要求：
        1. 如果片段中有相关信息，请结合小说内容给出详细、准确的回答
        2. 可以综合多个片段的内容，提供完整的答案
        3. 如果片段中没有相关信息，请如实告知用户
        4. 回答要准确，符合小说的情节和人物设定
        5. 可以引用原文内容来支持你的回答
        
        AI 助手的回答:`;

  console.log("\n【AI 回答】");
  const response = await model.invoke(prompt);
  const text = messageContentToString(response.content);
  console.log(text);
  console.log("\n");

  return text;
}

async function main(): Promise<void> {
  console.log("连接到 Milvus...");
  await client.connectPromise;
  console.log("✓ 已连接\n");

  try {
    await client.loadCollection({ collection_name: COLLECTION_NAME });
    console.log("✓ 集合已加载\n");
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "";
    if (!message.includes("already loaded")) {
      throw error;
    }
    console.log("✓ 集合已处于加载状态\n");
  }

  await answerEbookQuestion("鸠摩智会什么武功？", 5);
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error("错误:", message);
});
