import { Column, Entity, ManyToOne, PrimaryGeneratedColumn } from 'typeorm';
import { Cart } from './cart.entity';
import { ProductVariant } from '../catalog/product-variant.entity';

@Entity('cart_items')
export class CartItem {
  @PrimaryGeneratedColumn() id: number;
  @ManyToOne(() => Cart, (c) => c.items, { onDelete: 'CASCADE' }) cart: Cart;
  @ManyToOne(() => ProductVariant, { eager: true, onDelete: 'CASCADE' }) variant: ProductVariant;
  @Column({ default: 1 }) quantity: number;
}
