import { Entity, ManyToOne, PrimaryGeneratedColumn, Unique } from 'typeorm';
import { User } from '../users/user.entity';
import { Product } from '../catalog/product.entity';

@Entity('wishlists')
@Unique(['user', 'product'])
export class Wishlist {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => User, { onDelete: 'CASCADE' }) user: User;
  @ManyToOne(() => Product, { onDelete: 'CASCADE', eager: true }) product: Product;
}
