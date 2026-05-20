import { Column, Entity, Index, ManyToOne, PrimaryGeneratedColumn } from 'typeorm';
import { Product } from './product.entity';

@Entity('product_variants')
@Index(['sku'], { unique: true })
export class ProductVariant {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Product, (p) => p.variants, { onDelete: 'CASCADE' }) product: Product;
  @Column({ length: 60, nullable: true }) color: string;
  @Column({ length: 20, nullable: true }) size: string;
  @Column({ length: 80 }) sku: string;
  @Column({ default: 0 }) stock: number;
  @Column({ name: 'price_delta', type: 'decimal', precision: 12, scale: 2, default: 0 })
  priceDelta: string;
}
