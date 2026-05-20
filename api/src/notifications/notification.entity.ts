import {
  Column,
  CreateDateColumn,
  Entity,
  ManyToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { User } from '../users/user.entity';
import { NotificationType } from '../common/enums';

@Entity('notifications')
export class Notification {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => User, { onDelete: 'CASCADE' }) user: User;
  @Column({ length: 255 }) title: string;
  @Column({ type: 'text' }) content: string;
  @Column({ type: 'enum', enum: NotificationType, default: NotificationType.SYSTEM })
  type: NotificationType;
  @Column({ name: 'is_read', default: false }) isRead: boolean;
  @Column({ name: 'ref_type', length: 50, nullable: true }) refType: string;
  @Column({ name: 'ref_id', nullable: true }) refId: number;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
}
