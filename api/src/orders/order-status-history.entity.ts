import {
  Column,
  CreateDateColumn,
  Entity,
  ManyToOne,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { Order, OrderStatus } from './order.entity';
import { User } from '../users/user.entity';

@Entity('order_status_history')
export class OrderStatusHistory {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Order, { onDelete: 'CASCADE' }) order: Order;
  @Column({ name: 'old_status', type: 'enum', enum: OrderStatus, nullable: true })
  oldStatus: OrderStatus | null;
  @Column({ name: 'new_status', type: 'enum', enum: OrderStatus })
  newStatus: OrderStatus;
  @ManyToOne(() => User, { onDelete: 'SET NULL', nullable: true })
  changedBy: User;
  @Column({ length: 500, nullable: true }) note: string;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;
}
