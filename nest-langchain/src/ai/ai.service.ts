import { Inject, Injectable } from '@nestjs/common';
import { ChatOpenAI } from '@langchain/openai';
import {
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
} from '@langchain/core/messages';
import { UIMessage } from 'ai';
import { toBaseMessages, toUIMessageStream } from '@ai-sdk/langchain';
import { createAgent } from 'langchain';
import {
  AGENT_RECURSION_LIMIT,
  AI_CHAT_STREAM_SYSTEM_PROMPT,
  AI_CHAT_SYSTEM_PROMPT,
  AI_UI_CHAT_SYSTEM_PROMPT,
} from './ai-system-prompt';

type ChatAgent = ReturnType<typeof createAgent>;

function extractLastAssistantText(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.type !== 'ai') {
      continue;
    }

    const { content } = msg;
    if (typeof content === 'string') {
      return content;
    }

    if (Array.isArray(content)) {
      return content
        .map((block) =>
          typeof block === 'string'
            ? block
            : block.type === 'text'
              ? block.text
              : '',
        )
        .join('');
    }
  }

  return '';
}

function isAiTextChunk(message: unknown): message is AIMessageChunk {
  if (!message || typeof message !== 'object') {
    return false;
  }

  const chunk = message as AIMessageChunk;
  if (chunk.type !== 'ai') {
    return false;
  }

  return (chunk.tool_call_chunks?.length ?? 0) === 0;
}

function chunkText(message: AIMessageChunk): string {
  if (typeof message.content === 'string') {
    return message.content;
  }

  if (!Array.isArray(message.content)) {
    return '';
  }

  return message.content
    .map((block) =>
      typeof block === 'string'
        ? block
        : block.type === 'text'
          ? block.text
          : '',
    )
    .join('');
}

@Injectable()
export class AiService {
  private readonly chatAgent: ChatAgent;
  private readonly chatStreamAgent: ChatAgent;
  private readonly uiChatAgent: ChatAgent;

  constructor(
    @Inject('CHAT_MODEL') model: ChatOpenAI,
    @Inject('QUERY_USER_TOOL') queryUserTool: any,
    @Inject('SEND_MAIL_TOOL') sendMailTool: any,
    @Inject('WEB_SEARCH_TOOL') webSearchTool: any,
    @Inject('DB_USERS_CRUD_TOOL') dbUsersCrudTool: any,
    @Inject('TIME_NOW_TOOL') timeNowTool: any,
    @Inject('CRON_JOB_TOOL') cronJobTool: any,
  ) {
    const tools = [
      queryUserTool,
      sendMailTool,
      webSearchTool,
      dbUsersCrudTool,
      timeNowTool,
      cronJobTool,
    ];

    this.chatAgent = createAgent({
      model,
      tools,
      systemPrompt: AI_CHAT_SYSTEM_PROMPT,
    });

    this.chatStreamAgent = createAgent({
      model,
      tools,
      systemPrompt: AI_CHAT_STREAM_SYSTEM_PROMPT,
    });

    this.uiChatAgent = createAgent({
      model,
      tools,
      systemPrompt: AI_UI_CHAT_SYSTEM_PROMPT,
    });
  }

  async runChain(query: string): Promise<string> {
    const result = await this.chatAgent.invoke(
      { messages: [new HumanMessage(query)] },
      { recursionLimit: AGENT_RECURSION_LIMIT },
    );

    const messages = (result as { messages: BaseMessage[] }).messages ?? [];
    return extractLastAssistantText(messages);
  }

  async *runChainStream(query: string): AsyncIterable<string> {
    const stream = await this.chatStreamAgent.stream(
      { messages: [new HumanMessage(query)] },
      { streamMode: 'messages', recursionLimit: AGENT_RECURSION_LIMIT },
    );

    for await (const event of stream) {
      const message = Array.isArray(event) ? event[0] : event;
      if (!isAiTextChunk(message)) {
        continue;
      }

      const text = chunkText(message);
      if (text) {
        yield text;
      }
    }
  }

  async stream(messages: UIMessage[]) {
    const lcMessages = await toBaseMessages(messages);
    const lgStream = await this.uiChatAgent.stream(
      { messages: lcMessages },
      {
        streamMode: ['messages', 'values'],
        recursionLimit: AGENT_RECURSION_LIMIT,
      },
    );
    return toUIMessageStream(lgStream);
  }
}
