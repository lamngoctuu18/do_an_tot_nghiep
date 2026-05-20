import {
  Column,
  CreateDateColumn,
  Entity,
  ManyToOne,
  OneToMany,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { User } from '../users/user.entity';
import { OrderItem } from './order-item.entity';
import { Seller } from '../sellers/seller.entity';

export enum OrderStatus {
  PENDING = 'PENDING',
  CONFIRMED = 'CONFIRMED',
  PACKING = 'PACKING',
  SHIPPING = 'SHIPPING',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED',
  // legacy
  PAID = 'PAID',
  DELIVERED = 'DELIVERED',
}

export enum PaymentStatus {
  UNPAID = 'UNPAID',
  PAID = 'PAID',
  REFUNDED = 'REFUNDED',
  FAILED = 'FAILED',
}

export enum PaymentMethod {
  COD = 'COD',
  VNPAY = 'VNPAY',
}

@Entity('orders')
export class Order {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => User, { onDelete: 'SET NULL', nullable: true }) user: User;
  @ManyToOne(() => Seller, { onDelete: 'SET NULL', nullable: true }) shop: Seller;
  @Column({ unique: true, length: 32 }) code: string;
  @Column({ type: 'decimal', precision: 12, scale: 2 }) subtotal: string;
  @Column({ name: 'shipping_fee', type: 'decimal', precision: 12, scale: 2, default: 0 })
  shippingFee: string;
  @Column({ type: 'decimal', precision: 12, scale: 2, default: 0 }) discount: string;
  @Column({ type: 'decimal', precision: 12, scale: 2 }) total: string;
  @Column({ type: 'enum', enum: OrderStatus, default: OrderStatus.PENDING })
  status: OrderStatus;
  @Column({
    name: 'payment_status',
    type: 'enum',
    enum: PaymentStatus,
    default: PaymentStatus.UNPAID,
  })
  paymentStatus: PaymentStatus;
  @Column({ name: 'payment_method', type: 'enum', enum: PaymentMethod, default: PaymentMethod.COD })
  paymentMethod: PaymentMethod;
  @Column({ name: 'address_snapshot', type: 'json' }) addressSnapshot: any;
  @Column({ type: 'text', nullable: true }) note: string;
  @Column({ name: 'cancel_reason', type: 'text', nullable: true }) cancelReason: string;
  @CreateDateColumn({ name: 'placed_at' }) placedAt: Date;

  @OneToMany(() => OrderItem, (i) => i.order, { cascade: true }) items: OrderItem[];
}
