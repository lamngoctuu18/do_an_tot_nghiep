import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { User } from '../users/user.entity';
import { Seller } from '../sellers/seller.entity';
import { Order } from '../orders/order.entity';
import { Product } from '../catalog/product.entity';
import { Category } from '../catalog/category.entity';
import { Review } from '../reviews/review.entity';
import { TryonSession } from '../tryon/tryon-session.entity';
import { AdminController } from './admin.controller';
import { NotificationsModule } from '../notifications/notifications.module';

@Module({
  imports: [
    TypeOrmModule.forFeature([User, Seller, Order, Product, Category, Review, TryonSession]),
    NotificationsModule,
  ],
  controllers: [AdminController],
})
export class AdminModule {}
