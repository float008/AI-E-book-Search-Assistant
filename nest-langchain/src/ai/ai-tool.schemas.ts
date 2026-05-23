import z from 'zod';

export const queryUserSchema = z.object({
  userId: z.string().describe('用户ID，例如： 001, 002, 003'),
});

export const sendMailArgsSchema = z.object({
  to: z.string().email().describe('收件人邮箱'),
  subject: z.string().describe('邮件主题'),
  text: z.string().optional().describe('纯文本内容，可选'),
  html: z.string().optional().describe('HTML内容，可选'),
});

export const webSearchArgsSchema = z.object({
  query: z.string().min(1).describe('搜索关键词'),
  count: z.number().min(1).max(10).describe('搜索结果数量'),
});

export type QueryUserArgs = z.infer<typeof queryUserSchema>;
export type SendMailArgs = z.infer<typeof sendMailArgsSchema>;
export type WebSearchArgs = z.infer<typeof webSearchArgsSchema>;
