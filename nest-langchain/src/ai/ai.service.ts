import { StringOutputParser } from '@langchain/core/output_parsers';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Runnable } from '@langchain/core/runnables';
import { ChatOpenAI } from '@langchain/openai';
import { Inject, Injectable } from '@nestjs/common';

@Injectable()
export class AiService {
  private readonly chain: Runnable<{ query: string }, string>;

  constructor(@Inject('CHAT_MODEL') private readonly model: ChatOpenAI) {
    const prompt = ChatPromptTemplate.fromTemplate('请回答问题：{query}');
    this.chain = prompt.pipe(model).pipe(new StringOutputParser());
  }

  async runChain(query: string): Promise<string> {
    return this.chain.invoke({ query });
  }

  async *streamChain(query: string): AsyncGenerator<string> {
    const stream = await this.chain.stream({ query });

    for await (const chunk of stream) {
      yield chunk;
    }
  }
}
