import { Injectable } from '@nestjs/common';

type User = {
  id: string;
  name: string;
  role: number;
  email: string;
};

@Injectable()
export class UserService {
  private readonly users = new Map<string, User>([
    [
      '001',
      { id: '001', name: '刘备', role: 1, email: 'zhangsan@example.com' },
    ],
    ['002', { id: '002', name: '关羽', role: 2, email: 'lisi@example.com' }],
    ['003', { id: '003', name: '张飞', role: 3, email: 'wangwu@example.com' }],
  ]);

  findAll(): User[] {
    return Array.from(this.users.values());
  }

  findOne(id: string): User | undefined {
    return this.users.get(id);
  }

  create(user: User) {
    this.users.set(user.id, user);
    return user;
  }

  update(id: string, partial: Partial<Omit<User, 'id'>>): User | undefined {
    const existing = this.users.get(id);
    if (!existing) {
      return undefined;
    }
    const updated: User = { ...existing, ...partial, id: existing.id };
    this.users.set(id, updated);
    return updated;
  }

  remove(id: string): boolean {
    return this.users.delete(id);
  }
}
