import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  ManyToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { User } from '../users/user.entity';

@Entity('refresh_tokens')
export class RefreshToken {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => User, { onDelete: 'CASCADE' }) user: User;
  @Index({ unique: true })
  @Column({ length: 200 })
  token: string;
  @Column({ name: 'expires_at', type: 'datetime' }) expiresAt: Date;
  @Column({ default: false }) revoked: boolean;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
}
