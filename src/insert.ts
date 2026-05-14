import "dotenv/config";
import { parse } from "node:path";
import {
  MilvusClient,
  DataType,
  MetricType,
  IndexType,
} from "@zilliz/milvus2-sdk-node";
import { OpenAIEmbeddings } from "@langchain/openai";
import { EPubLoader } from "@langchain/community/document_loaders/fs/epub";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const COLLECTION_NAME = "ebook_collection";
const VECTOR_DIM = 1024;
const CHUNK_SIZE = 500;
const EPUB_FILE = "./天龙八部.epub";
const BOOK_NAME = parse(EPUB_FILE).name;

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

async function getEmbedding(text: string): Promise<number[]> {
  return embeddings.embedQuery(text);
}

async function removeExistingChunksForBook(
  bookId: number | string,
): Promise<void> {
  const has = await client.hasCollection({
    collection_name: COLLECTION_NAME,
  });
  if (!has.value) {
    return;
  }

  const bid = String(bookId);
  console.log(`\n清除 book_id=${bid} 的旧数据（若存在）...`);

  const del = await client.delete({
    collection_name: COLLECTION_NAME,
    filter: `book_id == "${bid}"`,
  });

  if (del.status?.error_code !== "Success" && del.status?.error_code !== 0) {
    throw new Error(del.status?.reason || String(del.status?.error_code));
  }

  await client.flushSync({ collection_names: [COLLECTION_NAME] });
  console.log("✓ 旧数据已清除（或本来无数据）\n");
}

async function ensureCollection(): Promise<void> {
  try {
    const hasCollection = await client.hasCollection({
      collection_name: COLLECTION_NAME,
    });

    console.log("hasCollection", hasCollection);

    if (!hasCollection.value) {
      console.log("创建集合");

      await client.createCollection({
        collection_name: COLLECTION_NAME,
        fields: [
          {
            name: "id",
            data_type: DataType.VarChar,
            max_length: 100,
            is_primary_key: true,
          },
          {
            name: "book_id",
            data_type: DataType.VarChar,
            max_length: 100,
          },
          { name: "book_name", data_type: DataType.VarChar, max_length: 200 },
          { name: "chapter_num", data_type: DataType.Int32 },
          { name: "index", data_type: DataType.Int32 },
          { name: "content", data_type: DataType.VarChar, max_length: 10000 },
          { name: "vector", data_type: DataType.FloatVector, dim: VECTOR_DIM },
        ],
      });

      console.log("✓ 集合创建成功");

      console.log("创建索引...");
      await client.createIndex({
        collection_name: COLLECTION_NAME,
        field_name: "vector",
        index_type: IndexType.IVF_FLAT,
        metric_type: MetricType.COSINE,
        params: { nlist: 1024 },
      });
      console.log("✓ 索引创建成功");
    }

    try {
      await client.loadCollection({ collection_name: COLLECTION_NAME });
      console.log("✓ 集合已加载");
    } catch {
      console.log("✓ 集合已处于加载状态");
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("创建集合时出错:", message);
    console.error(error);
  }
}

async function insertChunksBatch(
  chunks: string[],
  bookId: number | string,
  chapterNum: number,
): Promise<number> {
  try {
    if (chunks.length === 0) {
      return 0;
    }

    const bookIdStr = String(bookId);

    const insertData = await Promise.all(
      chunks.map(async (chunk, chunkIndex) => {
        const vector = await getEmbedding(chunk);

        return {
          id: `${bookIdStr}_${chapterNum}_${chunkIndex}`,
          book_id: bookIdStr,
          book_name: BOOK_NAME,
          chapter_num: chapterNum,
          index: chunkIndex,
          content: chunk,
          vector,
        };
      }),
    );

    const insertResult = await client.insert({
      collection_name: COLLECTION_NAME,
      data: insertData,
    });

    console.log("insertResult", insertResult);

    const cnt = insertResult.insert_cnt;
    return typeof cnt === "number" ? cnt : Number(cnt) || 0;
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`插入章节 ${chapterNum} 的数据时出错:`, message);
    console.error("错误详情:", error);
    return 0;
  }
}

async function loadAndProcessEPubStreaming(
  bookId: number | string,
): Promise<void> {
  await removeExistingChunksForBook(bookId);

  console.log(`\n开始加载 EPUB 文件: ${EPUB_FILE}`);

  const loader = new EPubLoader(EPUB_FILE, {
    splitChapters: true,
  });

  const document = await loader.load();
  console.log(`✓ 加载完成，共 ${document.length} 个章节\n`);

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: 50,
  });

  let totalInserted = 0;

  for (let chapterIndex = 0; chapterIndex < document.length; chapterIndex++) {
    const chapter = document[chapterIndex];
    const chapterContent = chapter.pageContent;

    console.log(`\n开始处理第 ${chapterIndex + 1} 章`);

    const chunks = await textSplitter.splitText(chapterContent);

    console.log(
      `✓ 第 ${chapterIndex + 1} 章拆分完成，共 ${chunks.length} 个分块`,
    );

    if (chunks.length === 0) {
      console.log("跳过空章节 \n");
      continue;
    }

    console.log("生成向量并插入中...");

    const insertedCount = await insertChunksBatch(
      chunks,
      bookId,
      chapterIndex + 1,
    );

    totalInserted += insertedCount;

    console.log(`已插入${insertedCount}条记录（累计：${totalInserted}） \n`);
  }

  console.log(`\n 总共插入${totalInserted}条记录\n`);
}

async function main(): Promise<void> {
  console.log("=".repeat(80));
  console.log("电子书处理程序");
  console.log("=".repeat(80));

  console.log("\n连接 Milvus 数据库...");
  await client.connectPromise;
  console.log("✓ 连接成功");

  const bookId = 1;

  await ensureCollection();

  await loadAndProcessEPubStreaming(bookId);

  console.log("=".repeat(80));
  console.log("处理完成！");
  console.log("=".repeat(80));
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  const stack = error instanceof Error ? error.stack : undefined;
  console.error("\n错误：", message);
  if (stack) console.error(stack);
  process.exit(1);
});
