import {
  Column,
  CreateDateColumn,
  Entity,
  ManyToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { User } from '../users/user.entity';
import { Product } from '../catalog/product.entity';

@Entity('tryon_sessions')
export class TryonSession {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => User, { onDelete: 'SET NULL', nullable: true }) user: User;
  @ManyToOne(() => Product, { onDelete: 'SET NULL', nullable: true }) product: Product;
  @Column({ name: 'person_url', length: 500, nullable: true }) personUrl: string;
  @Column({ name: 'cloth_url', length: 500, nullable: true }) clothUrl: string;
  @Column({ name: 'result_url', length: 500, nullable: true }) resultUrl: string;
  @Column({ length: 60, nullable: true }) backend: string;
  @Column({ name: 'info_text', type: 'text', nullable: true }) infoText: string;
  @Column({ length: 16, default: 'OK' }) status: string;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
}
