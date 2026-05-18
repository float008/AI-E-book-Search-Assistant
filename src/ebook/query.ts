import "dotenv/config";
import { MilvusClient, MetricType } from "@zilliz/milvus2-sdk-node";
import { OpenAIEmbeddings } from "@langchain/openai";

const COLLECTION_NAME = "ebook_collection";
const VECTOR_DIM = 1024;

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.OPENAI_EMBEDDINGS_MODEL,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  dimensions: VECTOR_DIM,
});

const client = new MilvusClient({
  address: "localhost:19530",
});

async function getEmbedding(text: string): Promise<number[]> {
  return embeddings.embedQuery(text);
}

/** Milvus search 返回的单条命中（与 output_fields 一致） */
interface EbookSearchHit {
  id: string;
  book_id: string;
  chapter_num: number;
  index: number;
  content: string;
  score: number;
}

async function main(): Promise<void> {
  console.log("Connecting to Milvus...");
  await client.connectPromise;
  console.log("✓ Connected\n");

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

  console.log("Searching for similar ebook content...");
  const query = "鸠摩智会什么武功？";

  const queryVector = await getEmbedding(query);
  const searchResult = await client.search({
    collection_name: COLLECTION_NAME,
    vector: queryVector,
    limit: 5,
    metric_type: MetricType.COSINE,
    output_fields: ["id", "book_id", "chapter_num", "index", "content"],
  });

  const results = searchResult.results as EbookSearchHit[];

  console.log(`Found ${results.length} results:\n`);

  results.forEach((item, index) => {
    console.log(`${index + 1}. [Score: ${item.score.toFixed(4)}]`);
    console.log(`   ID: ${item.id}`);
    console.log(`   Book ID: ${item.book_id}`);
    console.log(`   Chapter: 第 ${item.chapter_num} 章`);
    console.log(`   Index: ${item.index}`);
    console.log(`   Content: ${item.content}\n`);
  });
}

main().catch((error) => {
  console.error(error);
});
