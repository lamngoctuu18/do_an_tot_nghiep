import { Column, Entity, ManyToOne, PrimaryGeneratedColumn } from 'typeorm';
import { Product } from './product.entity';

@Entity('product_images')
export class ProductImage {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Product, (p) => p.images, { onDelete: 'CASCADE' }) product: Product;
  @Column({ length: 500 }) url: string;
  @Column({ default: 0 }) position: number;
}
