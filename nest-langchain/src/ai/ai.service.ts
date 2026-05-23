import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages';
import { Runnable } from '@langchain/core/runnables';
import { Tool, tool } from '@langchain/core/tools';
import { ChatOpenAI } from '@langchain/openai';
import { Inject, Injectable } from '@nestjs/common';
import z from 'zod';

const database = {
  users: [
    { id: '001', name: '张三', age: 20, email: 'zhangsan@example.com' },
    { id: '002', name: '李四', age: 21, email: 'lisi@example.com' },
    { id: '003', name: '王五', age: 22, email: 'wangwu@example.com' },
  ],
};

const queryUserSchema = z.object({
  userId: z.string().describe('用户ID，例如： 001, 002, 003'),
});

type QueryUserArgs = { userId: string };

const queryUserTool = tool(
  ({ userId }: QueryUserArgs) => {
    const user = database.users.find((user) => user.id === userId);
    if (!user) {
      return `用户ID：${userId}，不存在`;
    }
    return `用户ID：${user.id}，姓名：${user.name}，年龄：${user.age}，邮箱：${user.email}`;
  },
  {
    name: 'query_user',
    description:
      '查询用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）',
    schema: queryUserSchema,
  },
);

@Injectable()
export class AiService {
  private readonly modelWithTools: Runnable<BaseMessage[], AIMessage>;

  constructor(
    @Inject('CHAT_MODEL') private readonly model: ChatOpenAI,
    @Inject('QUERY_USER_TOOL') private readonly queryUserTool: Tool,
  ) {
    this.modelWithTools = model.bindTools([queryUserTool]);
  }

  async runChain(query: string): Promise<string> {
    const messages: BaseMessage[] = [
      new SystemMessage(
        '你是一个智能助手，可以在需要时调用工具（如 query_user）来查询用户信息，再用结果回答用户的问题。',
      ),
      new HumanMessage(query),
    ];

    while (true) {
      const aiMessage = await this.modelWithTools.invoke(messages);
      messages.push(aiMessage);

      const toolCalls = aiMessage.tool_calls || [];

      if (!toolCalls.length) {
        return aiMessage.content as string;
      }

      for (const toolCall of toolCalls) {
        const toolCallId = toolCall.id || '';
        const toolName = toolCall.name;

        if (toolName === 'query_user') {
          const args = queryUserSchema.parse(toolCall.args);
          const result = (await this.queryUserTool.invoke({
            userId: args.userId,
          })) as string;

          messages.push(
            new ToolMessage({
              tool_call_id: toolCallId,
              name: toolName,
              content: result,
            }),
          );
        }
      }
    }
  }

  async *streamChain(query: string): AsyncGenerator<string> {
    const messages: BaseMessage[] = [
      new SystemMessage(
        '你是一个智能助手，可以在需要时调用工具（如 query_user）来查询用户信息，再用结果回答用户的问题。',
      ),
      new HumanMessage(query),
    ];

    while (true) {
      const stream = await this.modelWithTools.stream(messages);

      let fullAIMessage: AIMessageChunk | null = null;

      for await (const chunk of stream as AsyncIterable<AIMessageChunk>) {
        fullAIMessage = fullAIMessage ? fullAIMessage.concat(chunk) : chunk;

        const hasToolCallChunk =
          !!fullAIMessage.tool_call_chunks &&
          fullAIMessage.tool_call_chunks.length > 0;

        if (!hasToolCallChunk && chunk.content) {
          yield chunk.content as string;
        }
      }

      if (!fullAIMessage) {
        return;
      }

      messages.push(fullAIMessage);

      const toolCalls = fullAIMessage.tool_calls || [];

      if (!toolCalls.length) {
        return;
      }

      for (const toolCall of toolCalls) {
        const toolCallId = toolCall.id || '';
        const toolName = toolCall.name;

        if (toolName === 'query_user') {
          const args = queryUserSchema.parse(toolCall.args);

          const result = (await this.queryUserTool.invoke(args)) as string;

          messages.push(
            new ToolMessage({
              tool_call_id: toolCallId,
              name: toolName,
              content: result,
            }),
          );
        }
      }
    }
  }
}
