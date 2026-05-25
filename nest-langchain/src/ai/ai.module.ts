import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import { UserService } from './user.service';
import { tool } from '@langchain/core/tools';
import { MailerService } from '@nestjs-modules/mailer';
import { runBochaWebSearch } from './bocha-web-search';
import {
  queryUserSchema,
  sendMailArgsSchema,
  webSearchArgsSchema,
  type QueryUserArgs,
  type SendMailArgs,
  type WebSearchArgs,
} from './ai-tool.schemas';
import { UsersModule } from '../users/users.module';
import { UsersService } from '../users/users.service';
import { User } from '../users/entities/user.entity';
import z from 'zod';

const BOCHA_DEFAULT_URL = 'https://api.bochaai.com/v1/web-search';

@Module({
  imports: [UsersModule],
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
        return tool(
          ({ userId }: QueryUserArgs) => {
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
    {
      provide: 'SEND_MAIL_TOOL',
      useFactory: (
        mailerService: MailerService,
        configService: ConfigService,
      ) => {
        return tool(
          async ({ to, subject, text, html }: SendMailArgs) => {
            const from =
              configService.get<string>('MAIL_FROM') ??
              configService.get<string>('MAIL_USER');

            await mailerService.sendMail({
              to,
              subject,
              text: text ?? '（无文本内容）',
              html,
              from,
            });

            return `邮件发送成功，发送人：${from}，收件人：${to}，主题：${subject}`;
          },
          {
            name: 'send_mail',
            description:
              '发送邮件。输入收件人邮箱、邮件主题、纯文本内容、HTML内容，发送邮件',
            schema: sendMailArgsSchema,
          },
        );
      },
      inject: [MailerService, ConfigService],
    },
    {
      provide: 'WEB_SEARCH_TOOL',
      useFactory: (configService: ConfigService) => {
        return tool(
          async ({ query, count }: WebSearchArgs) => {
            const apiKey = configService.get<string>('BOCHA_API_KEY');
            if (!apiKey) {
              return 'Bocha Web Search 的 API Key 未配置（环境变量 BOCHA_API_KEY）。';
            }

            const baseUrl =
              configService.get<string>('BOCHA_BASE_URL') ?? BOCHA_DEFAULT_URL;

            return runBochaWebSearch(apiKey, baseUrl, query, count);
          },
          {
            name: 'web_search',
            description: '搜索网络信息',
            schema: webSearchArgsSchema,
          },
        );
      },
      inject: [ConfigService],
    },
    {
      provide: 'DB_USERS_CURD_TOOL',
      useFactory: (usersService: UsersService) => {
        const dbUsersCrudArgsSchema = z.object({
          action: z
            .enum(['create', 'list', 'get', 'update', 'delete'])
            .describe('要执行的操作：create、list、get、update、delete'),
          id: z
            .number()
            .int()
            .positive()
            .optional()
            .describe('用户 ID（get / update / delete 时需要）'),
          name: z
            .string()
            .min(1)
            .max(50)
            .optional()
            .describe('用户姓名（create 或 update 时可用）'),
          email: z
            .string()
            .email()
            .max(50)
            .optional()
            .describe('用户邮箱（create 或 update 时可用）'),
        });

        return tool(
          async ({
            action,
            id,
            name,
            email,
          }: {
            action: 'create' | 'list' | 'get' | 'update' | 'delete';
            id?: number;
            name?: string;
            email?: string;
          }) => {
            switch (action) {
              case 'create': {
                if (!name || !email) {
                  return '创建用户需要同时提供 name 和 email。';
                }
                const created: User = await usersService.create({
                  name,
                  email,
                });
                return `已创建用户：ID=${created.id}，姓名=${created.name}，邮箱=${created.email}`;
              }
              case 'list': {
                const users = await usersService.findAll();
                if (!users.length) {
                  return '数据库中还没有任何用户记录。';
                }
                const lines = users
                  .map((u) => `ID=${u.id}，姓名=${u.name}，邮箱=${u.email}`)
                  .join('\n');
                return `当前数据库 users 表中的用户列表：\n${lines}`;
              }
              case 'get': {
                if (!id) {
                  return '查询单个用户需要提供 id。';
                }
                try {
                  const user = await usersService.findOne(id);
                  return `用户信息：ID=${user.id}，姓名=${user.name}，邮箱=${user.email}`;
                } catch {
                  return `ID 为 ${id} 的用户在数据库中不存在。`;
                }
              }
              case 'update': {
                if (!id) {
                  return '更新用户需要提供 id。';
                }
                const payload: { name?: string; email?: string } = {};
                if (name !== undefined) payload.name = name;
                if (email !== undefined) payload.email = email;
                if (!Object.keys(payload).length) {
                  return '未提供需要更新的字段（name 或 email），本次不执行更新。';
                }
                try {
                  const updated = await usersService.update(id, payload);
                  return `已更新用户：ID=${updated.id}，姓名=${updated.name}，邮箱=${updated.email}`;
                } catch {
                  return `ID 为 ${id} 的用户在数据库中不存在。`;
                }
              }
              case 'delete': {
                if (!id) {
                  return '删除用户需要提供 id。';
                }
                try {
                  const removed = await usersService.remove(id);
                  return `已删除用户：ID=${removed.id}，姓名=${removed.name}，邮箱=${removed.email}`;
                } catch {
                  return `ID 为 ${id} 的用户在数据库中不存在，无需删除。`;
                }
              }
              default:
                return `不支持的操作: ${String(action)}`;
            }
          },
          {
            name: 'db_users_crud',
            description:
              '对数据库 users 表执行增删改查操作。通过 action 字段选择 create/list/get/update/delete，并按需提供 id、name、email 等参数。',
            schema: dbUsersCrudArgsSchema,
          },
        );
      },
      inject: [UsersService],
    },
  ],
})
export class AiModule {}
