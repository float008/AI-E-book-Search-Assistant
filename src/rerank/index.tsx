import { DashScopeRerank } from "./dashScopeRerank.ts";
import "dotenv/config";
import { Document } from "@langchain/core/documents";

const main = async () => {
  const compressor = new DashScopeRerank({
    apiKey: process.env.OPENAI_API_KEY ?? "",
    model: process.env.OPENAI_RERANK_MODEL ?? "qwen3-rerank",
    topN: 3,
    baseUrl: process.env.OPENAI_RERANK_URL,
  });

  const query = "什么是文本排序模型";

  const docs = [
    new Document({ pageContent: "预训练模型是用于文本排序的模型" }),
    new Document({ pageContent: "量子计算是计算科学的一个前沿领域" }),
    new Document({ pageContent: "人工智能是计算机科学的一个前沿领域" }),
  ];

  const ranked = await compressor.compressDocuments(docs, query);
  console.log("重排后顺序:");
  ranked.forEach((doc, i) => console.log(`${i + 1}. ${doc.pageContent}`));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
