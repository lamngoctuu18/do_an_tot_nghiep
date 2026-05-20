import {
  Column,
  CreateDateColumn,
  Entity,
  ManyToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { Product } from '../catalog/product.entity';
import { User } from '../users/user.entity';
import { ReviewStatus } from '../common/enums';

@Entity('reviews')
export class Review {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Product, { onDelete: 'CASCADE' }) product: Product;
  @ManyToOne(() => User, { onDelete: 'SET NULL', nullable: true }) user: User;
  @Column() rating: number;
  @Column({ type: 'text', nullable: true }) comment: string;
  @Column({ type: 'json', nullable: true }) images: string[];
  @Column({ type: 'enum', enum: ReviewStatus, default: ReviewStatus.VISIBLE })
  status: ReviewStatus;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
}
