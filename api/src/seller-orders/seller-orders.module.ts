import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Order } from '../orders/order.entity';
import { OrderStatusHistory } from '../orders/order-status-history.entity';
import { Seller } from '../sellers/seller.entity';
import { Product } from '../catalog/product.entity';
import { ProductVariant } from '../catalog/product-variant.entity';
import { TryonSession } from '../tryon/tryon-session.entity';
import { SellerOrdersController } from './seller-orders.controller';
import { NotificationsModule } from '../notifications/notifications.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([Order, OrderStatusHistory, Seller, Product, ProductVariant, TryonSession]),
    NotificationsModule,
  ],
  controllers: [SellerOrdersController],
})
export class SellerOrdersModule {}
