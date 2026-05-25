import { Inject, Injectable } from '@nestjs/common';
import { tool } from '@langchain/core/tools';
import { queryUserSchema, type QueryUserArgs } from '../ai/ai-tool.schemas';
import { UsersService } from '../users/users.service';

function formatUser(user: {
  id: number;
  name: string;
  email: string;
  createAt: Date;
  updateAt: Date;
}): string {
  return `用户信息：
- ID: ${user.id}
- 姓名: ${user.name}
- 邮箱: ${user.email}
- 创建时间: ${user.createAt.toISOString()}
- 更新时间: ${user.updateAt.toISOString()}`;
}

@Injectable()
export class QueryUserToolService {
  readonly tool;

  @Inject(UsersService)
  private readonly usersService: UsersService;

  constructor() {
    this.tool = tool(
      async ({ userId }: QueryUserArgs) => {
        const id = Number(userId);
        if (!Number.isInteger(id) || id <= 0) {
          return `用户 ID「${userId}」无效，请提供正整数（与 users 表主键一致）。`;
        }

        try {
          const user = await this.usersService.findOne(id);
          return formatUser(user);
        } catch {
          const users = await this.usersService.findAll();
          const availableIds = users.map((u) => u.id).join(', ');
          return `用户 ID ${userId} 不存在。${availableIds ? `可用的 ID: ${availableIds}` : '数据库中暂无用户'}`;
        }
      },
      {
        name: 'query_user',
        description:
          '查询 MySQL users 表中的单个用户。输入用户 ID（数字，如 1、2），返回姓名、邮箱等信息。',
        schema: queryUserSchema,
      },
    );
  }
}
