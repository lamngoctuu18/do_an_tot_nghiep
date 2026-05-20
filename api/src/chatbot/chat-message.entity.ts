import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  ManyToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { User } from '../users/user.entity';

export enum ChatRole {
  USER = 'USER',
  BOT = 'BOT',
}

@Entity('chat_messages')
@Index(['sessionId'])
export class ChatMessage {
  @PrimaryGeneratedColumn() id: number;

  @ManyToOne(() => User, { onDelete: 'CASCADE', nullable: true })
  user: User;

  @Column({ name: 'session_id', length: 64 })
  sessionId: string;

  @Column({ type: 'enum', enum: ChatRole })
  role: ChatRole;

  @Column({ type: 'text' })
  content: string;

  @Column({ type: 'json', nullable: true })
  metadata: any;

  @CreateDateColumn({ name: 'created_at' })
  createdAt: Date;
}
