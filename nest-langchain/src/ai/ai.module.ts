import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import { UserService } from './user.service';
import z from 'zod';
import { tool } from '@langchain/core/tools';

@Module({
  controllers: [AiController],
  providers: [
    AiService,
    UserService,
    {
      provide: 'CHAT_MODEL',
      useFactory: (configService: ConfigService) => {
        return new ChatOpenAI({
          model: configService.get<string>('OPENAI_BASE_MODEL'),
          temperature: 0,
          apiKey: configService.get<string>('OPENAI_API_KEY'),
          configuration: {
            baseURL: configService.get<string>('OPENAI_BASE_URL'),
          },
        });
      },
      inject: [ConfigService],
    },
    {
      provide: 'QUERY_USER_TOOL',
      useFactory: (userService: UserService) => {
        const queryUserSchema = z.object({
          userId: z.string().describe('用户ID，例如： 001, 002, 003'),
        });

        return tool(
          ({ userId }: { userId: string }) => {
            const users = userService.findOne(userId);

            if (!users) {
              const availableUsers = userService
                .findAll()
                .map((user) => user.id)
                .join(', ');

              return `用户ID：${userId}，不存在。可用的用户ID：${availableUsers}`;
            }

            return `用户ID：${users.id}，姓名：${users.name}，邮箱：${users.email}`;
          },
          {
            name: 'query_user',
            description:
              '查询数据库中的用户信息。输入用户 ID，返回该用户的详细信息（姓名、邮箱、角色）',
            schema: queryUserSchema,
          },
        );
      },
      inject: [UserService],
    },
  ],
})
export class AiModule {}
