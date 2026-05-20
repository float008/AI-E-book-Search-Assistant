/**
 * OpenEvals 内置 RAG 指标
 */
import {
  createLLMAsJudge,
  RAG_GROUNDEDNESS_PROMPT,
  RAG_HELPFULNESS_PROMPT,
  RAG_RETRIEVAL_RELEVANCE_PROMPT,
} from "openevals";
import { ChatOpenAI } from "@langchain/openai";

const judge = new ChatOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  configuration: { baseURL: process.env.OPENAI_BASE_URL },
  model: process.env.OPENAI_BASE_MODEL,
  temperature: 0,
});

// RAG_GROUNDEDNESS_PROMPT —— 忠实度：答案是否被检索上下文支撑，有无幻觉
const ragGroundednessJudge = createLLMAsJudge({
  prompt: RAG_GROUNDEDNESS_PROMPT,
  feedbackKey: "rag_groundedness",
  judge,
  continuous: true,
});

// RAG_HELPFULNESS_PROMPT —— 回答有用性：是否切题、是否答非所问
const ragHelpfulnessJudge = createLLMAsJudge({
  prompt: RAG_HELPFULNESS_PROMPT,
  feedbackKey: "rag_helpfulness",
  judge,
  continuous: true,
});

// RAG_RETRIEVAL_RELEVANCE_PROMPT —— 检索相关性：召回片段与问题是否相关
const ragRetrievalRelevanceJudge = createLLMAsJudge({
  prompt: RAG_RETRIEVAL_RELEVANCE_PROMPT,
  feedbackKey: "rag_retrieval_relevance",
  judge,
  continuous: true,
});

export async function ragGroundednessEvaluator({ outputs }: any) {
  return ragGroundednessJudge({
    context: { documents: outputs.context },
    outputs: { answer: outputs.answer },
  });
}

export async function ragHelpfulnessEvaluator({ inputs, outputs }: any) {
  return ragHelpfulnessJudge({ inputs, outputs: { answer: outputs.answer } });
}

export async function ragRetrievalRelevanceEvaluator({ inputs, outputs }: any) {
  return ragRetrievalRelevanceJudge({
    inputs,
    context: { documents: outputs.context },
  });
}

export const ragEvaluators = [
  ragGroundednessEvaluator,
  ragHelpfulnessEvaluator,
  ragRetrievalRelevanceEvaluator,
];
