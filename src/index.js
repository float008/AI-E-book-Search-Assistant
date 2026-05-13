import "dotenv/config";
import { parse } from "path";
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
const CHUNK_SIZE = 500; // 拆分到 500 个字符
const EPUB_FILE = "./天龙八部.epub";
const BOOK_NAME = parse(EPUB_FILE).name;

// 初始化 Milvus 客户端
const client = new MilvusClient({
  address: "localhost:19530",
});

//初始化向量模型
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: process.env.OPENAI_EMBEDDINGS_MODEL,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
  dimensions: VECTOR_DIM,
});

async function getEmbedding(text) {
  const embedding = await embeddings.embedQuery(text);
  return embedding;
}

/** 同一本书重复导入前，按 book_id 删除旧分片，避免主键与数据重复 */
async function removeExistingChunksForBook(bookId) {
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

// 确保集合存在
async function ensureCollection() {
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

      // 创建索引
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

    // 确保集合已加载
    try {
      await client.loadCollection({ collection_name: COLLECTION_NAME });
      console.log("✓ 集合已加载");
    } catch (error) {
      console.log("✓ 集合已处于加载状态");
    }
  } catch (error) {
    console.error("创建集合时出错:", error.message);
    console.error(error);
  }
}

/**
 * 将文档块批量插入到 Milvus
 */
async function insertChunksBatch(chunks, bookId, chapterNum) {
  try {
    if (chunks.length === 0) {
      return 0;
    }

    const bookIdStr = String(bookId);

    // 为每个文档生成向量并构建插入数据

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
          vector: vector,
        };
      }),
    );

    const insertResult = await client.insert({
      collection_name: COLLECTION_NAME,
      data: insertData,
    });

    console.log("insertResult", insertResult);

    return Number(insertResult.insert_cnt) || 0;
  } catch (error) {
    console.error(`插入章节 ${chapterNum} 的数据时出错:`, error.message);
    console.error("错误详情:", error);
    console.error(error);
    return 0;
  }
}

async function loadAndProcessEPubStreaming(bookId) {
  try {
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
  } catch (error) {
    console.error(`加载 EPUB 文件时出错:`, error.message);
    throw error;
  }
}

async function main() {
  try {
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
  } catch (error) {
    console.error("\n错误：", error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
