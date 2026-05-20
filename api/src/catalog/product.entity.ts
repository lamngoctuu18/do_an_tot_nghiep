import {
  Column,
  CreateDateColumn,
  Entity,
  Index,
  ManyToOne,
  OneToMany,
  PrimaryGeneratedColumn,
} from 'typeorm';
import { Seller } from '../sellers/seller.entity';
import { Category } from './category.entity';
import { ProductImage } from './product-image.entity';
import { ProductVariant } from './product-variant.entity';
import { ProductTryonAsset } from './product-tryon-asset.entity';

export enum ProductStatus {
  DRAFT = 'DRAFT',
  PENDING = 'PENDING',
  ACTIVE = 'ACTIVE',
  REJECTED = 'REJECTED',
  HIDDEN = 'HIDDEN',
}

@Entity('products')
@Index(['slug'], { unique: true })
export class Product {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Seller, { onDelete: 'CASCADE', nullable: true }) seller: Seller;
  @ManyToOne(() => Category, { onDelete: 'SET NULL', nullable: true }) category: Category;
  @Column({ length: 200 }) name: string;
  @Column({ length: 220 }) slug: string;
  @Column({ type: 'text', nullable: true }) description: string;
  @Column({ type: 'decimal', precision: 12, scale: 2 }) price: string;
  @Column({ name: 'original_price', type: 'decimal', precision: 12, scale: 2, nullable: true })
  originalPrice: string;
  @Column({ name: 'try_on_enabled', default: false }) tryOnEnabled: boolean;
  @Column({ type: 'enum', enum: ProductStatus, default: ProductStatus.ACTIVE })
  status: ProductStatus;
  @Column({ name: 'reject_reason', type: 'text', nullable: true }) rejectReason: string;
  @Column({ name: 'rating_avg', type: 'decimal', precision: 3, scale: 2, default: 0 })
  ratingAvg: string;
  @Column({ name: 'rating_count', default: 0 }) ratingCount: number;
  @Column({ name: 'sales_count', default: 0 }) salesCount: number;
  @Column({ length: 32, nullable: true }) badge: string;
  @CreateDateColumn({ name: 'created_at' }) createdAt: Date;

  @OneToMany(() => ProductImage, (i) => i.product, { cascade: true }) images: ProductImage[];
  @OneToMany(() => ProductVariant, (v) => v.product, { cascade: true }) variants: ProductVariant[];
  @OneToMany(() => ProductTryonAsset, (t) => t.product, { cascade: true }) tryonAssets: ProductTryonAsset[];
}
