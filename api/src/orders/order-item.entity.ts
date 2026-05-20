import { Column, Entity, ManyToOne, PrimaryGeneratedColumn } from 'typeorm';
import { Order } from './order.entity';
import { Product } from '../catalog/product.entity';
import { ProductVariant } from '../catalog/product-variant.entity';

@Entity('order_items')
export class OrderItem {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Order, (o) => o.items, { onDelete: 'CASCADE' }) order: Order;
  @ManyToOne(() => Product, { onDelete: 'SET NULL', nullable: true }) product: Product;
  @ManyToOne(() => ProductVariant, { onDelete: 'SET NULL', nullable: true }) variant: ProductVariant;
  @Column({ name: 'name_snapshot', length: 200 }) nameSnapshot: string;
  @Column({ name: 'price_snapshot', type: 'decimal', precision: 12, scale: 2 })
  priceSnapshot: string;
  @Column() quantity: number;
}
