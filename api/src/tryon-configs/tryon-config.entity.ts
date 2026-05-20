import {
  Column,
  CreateDateColumn,
  Entity,
  PrimaryGeneratedColumn,
  UpdateDateColumn,
} from 'typeorm';

@Entity('tryon_configs')
export class TryonConfig {
  @PrimaryGeneratedColumn() id: number;
  @Column({ length: 50 }) provider: string;
  @Column({ length: 50 }) mode: string;
  @Column({ name: 'is_enabled', default: true }) isEnabled: boolean;
  @Column({ name: 'max_daily_usage', default: 100 }) maxDailyUsage: number;
  @Column({ name: 'timeout_seconds', default: 120 }) timeoutSeconds: number;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
  @UpdateDateColumn({ name: 'updated_at' }) updatedAt: Date;
}
