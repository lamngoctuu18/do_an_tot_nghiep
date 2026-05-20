import { Column, Entity, ManyToOne, PrimaryGeneratedColumn } from 'typeorm';
import { Product } from './product.entity';

export enum ClothType {
  AUTO = 'auto',
  UPPER = 'upper',
  LOWER = 'lower',
  OVERALL = 'overall',
}

@Entity('product_tryon_assets')
export class ProductTryonAsset {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Product, (p) => p.tryonAssets, { onDelete: 'CASCADE' }) product: Product;
  @Column({ name: 'cloth_image_url', length: 500 }) clothImageUrl: string;
  @Column({ name: 'cloth_type', type: 'enum', enum: ClothType, default: ClothType.AUTO })
  clothType: ClothType;
}
