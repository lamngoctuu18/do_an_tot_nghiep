import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ServeStaticModule } from '@nestjs/serve-static';
import { join } from 'path';

import { AuthModule } from './auth/auth.module';
import { UsersModule } from './users/users.module';
import { AddressesModule } from './addresses/addresses.module';
import { CatalogModule } from './catalog/catalog.module';
import { CartModule } from './cart/cart.module';
import { OrdersModule } from './orders/orders.module';
import { PaymentsModule } from './payments/payments.module';
import { ReviewsModule } from './reviews/reviews.module';
import { WishlistModule } from './wishlist/wishlist.module';
import { SellersModule } from './sellers/sellers.module';
import { SellerOrdersModule } from './seller-orders/seller-orders.module';
import { AdminModule } from './admin/admin.module';
import { TryonModule } from './tryon/tryon.module';
import { TryonConfigsModule } from './tryon-configs/tryon-configs.module';
import { NotificationsModule } from './notifications/notifications.module';
import { ChatbotModule } from './chatbot/chatbot.module';
import { HealthController } from './health.controller';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true, envFilePath: '.env' }),
    TypeOrmModule.forRootAsync({
      inject: [ConfigService],
      useFactory: (cs: ConfigService) => ({
        type: 'mysql',
        host: cs.get('DB_HOST', 'localhost'),
        port: +cs.get('DB_PORT', 3306),
        username: cs.get('DB_USER', 'root'),
        password: cs.get<string>('DB_PASSWORD') ?? cs.get<string>('DB_PASS', ''),
        database: cs.get('DB_NAME', 'vton_shop'),
        entities: [join(__dirname, '**', '*.entity.{ts,js}')],
        synchronize: cs.get('NODE_ENV') !== 'production',
        charset: 'utf8mb4',
      }),
    }),
    ServeStaticModule.forRoot({
      rootPath: join(process.cwd(), 'uploads'),
      serveRoot: '/uploads',
    }),
    AuthModule,
    UsersModule,
    AddressesModule,
    CatalogModule,
    CartModule,
    OrdersModule,
    PaymentsModule,
    ReviewsModule,
    WishlistModule,
    SellersModule,
    SellerOrdersModule,
    AdminModule,
    TryonModule,
    TryonConfigsModule,
    NotificationsModule,
    ChatbotModule,
  ],
  controllers: [HealthController],
})
export class AppModule {}
