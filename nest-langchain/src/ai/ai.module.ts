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

const BOCHA_DEFAULT_URL = 'https://api.bochaai.com/v1/web-search';

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
  ],
})
export class AiModule {}
