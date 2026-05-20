import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Cart } from './cart.entity';
import { CartItem } from './cart-item.entity';
import { ProductVariant } from '../catalog/product-variant.entity';
import { CartController } from './cart.controller';

@Module({
  imports: [TypeOrmModule.forFeature([Cart, CartItem, ProductVariant])],
  controllers: [CartController],
  exports: [TypeOrmModule],
})
export class CartModule {}
