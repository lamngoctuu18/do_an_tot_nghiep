import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Order } from './order.entity';
import { OrderItem } from './order-item.entity';
import { OrderStatusHistory } from './order-status-history.entity';
import { Cart } from '../cart/cart.entity';
import { Address } from '../users/address.entity';
import { ProductVariant } from '../catalog/product-variant.entity';
import { Product } from '../catalog/product.entity';
import { OrdersController } from './orders.controller';
import { NotificationsModule } from '../notifications/notifications.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([
      Order,
      OrderItem,
      OrderStatusHistory,
      Cart,
      Address,
      ProductVariant,
      Product,
    ]),
    NotificationsModule,
  ],
  controllers: [OrdersController],
  exports: [TypeOrmModule],
})
export class OrdersModule {}
