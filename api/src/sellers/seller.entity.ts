import {
  Column,
  CreateDateColumn,
  Entity,
  JoinColumn,
  OneToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { User } from '../users/user.entity';

export enum SellerStatus {
  PENDING = 'PENDING',
  APPROVED = 'APPROVED',
  REJECTED = 'REJECTED',
  SUSPENDED = 'SUSPENDED',
}

@Entity('sellers')
export class Seller {
  @PrimaryGeneratedColumn() id: number;
  @OneToOne(() => User, (u) => u.seller, { onDelete: 'CASCADE' })
  @JoinColumn({ name: 'user_id' })
  user: User;
  @Column({ name: 'shop_name', length: 120 }) shopName: string;
  @Column({ unique: true, length: 140 }) slug: string;
  @Column({ name: 'logo_url', length: 500, nullable: true }) logoUrl: string;
  @Column({ name: 'banner_url', length: 500, nullable: true }) bannerUrl: string;
  @Column({ type: 'text', nullable: true }) description: string;
  @Column({ type: 'enum', enum: SellerStatus, default: SellerStatus.PENDING })
  status: SellerStatus;
  @Column({ name: 'reject_reason', type: 'text', nullable: true }) rejectReason: string;
  @Column({ name: 'commission_rate', type: 'decimal', precision: 5, scale: 2, default: 5 })
  commissionRate: string;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
}
