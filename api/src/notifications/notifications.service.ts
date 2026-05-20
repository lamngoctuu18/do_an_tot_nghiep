import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Notification } from './notification.entity';
import { NotificationType } from '../common/enums';

@Injectable()
export class NotificationsService {
  constructor(
    @InjectRepository(Notification) private repo: Repository<Notification>,
  ) {}

  create(
    userId: number,
    title: string,
    content: string,
    type: NotificationType = NotificationType.SYSTEM,
    refType?: string,
    refId?: number,
  ) {
    return this.repo.save(
      this.repo.create({
        user: { id: userId } as any,
        title,
        content,
        type,
        refType,
        refId,
      }),
    );
  }

  list(userId: number, isRead?: boolean) {
    const where: any = { user: { id: userId } };
    if (isRead !== undefined) where.isRead = isRead;
    return this.repo.find({ where, order: { createdAt: 'DESC' }, take: 200 });
  }

  async markRead(userId: number, id: number) {
    await this.repo.update({ id, user: { id: userId } as any }, { isRead: true });
    return { success: true };
  }

  async markAllRead(userId: number) {
    await this.repo.update({ user: { id: userId } as any, isRead: false }, { isRead: true });
    return { success: true };
  }
}
