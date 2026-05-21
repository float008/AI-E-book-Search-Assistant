import "dotenv/config";
import { BaseDocumentCompressor } from "@langchain/core/retrievers/document_compressors";
import { Document } from "@langchain/core/documents";

export class DashScopeRerank extends BaseDocumentCompressor {
  private apiKey: string;
  private model: string;
  private topN: number;
  private baseUrl: string;

  constructor({
    apiKey,
    model = "qwen3-rerank",
    topN = 3,
    baseUrl,
  }: {
    apiKey: string;
    model?: string;
    topN?: number;
    baseUrl?: string;
  }) {
    super();
    this.apiKey = apiKey;
    this.model = model;
    this.topN = topN;
    this.baseUrl = baseUrl ?? "";
  }

  async compressDocuments(documents: Document[], query: string) {
    const res = await fetch(this.baseUrl, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.model,
        input: {
          query,
          documents: documents.map((d) => d.pageContent),
        },
        parameters: {
          return_documents: false,
          top_n: this.topN,
        },
      }),
    });

    const json: any = await res.json();
    if (!res.ok) {
      throw new Error(
        `DashScope rerank ${res.status}: ${JSON.stringify(json)}`,
      );
    }

    const results = json?.output?.results;
    if (!Array.isArray(results)) {
      throw new Error(`unexpected rerank response: ${JSON.stringify(json)}`);
    }

    return results.map((item) => documents[item.index]);
  }
}
