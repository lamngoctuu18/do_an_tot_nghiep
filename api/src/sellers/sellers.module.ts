import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Seller } from './seller.entity';
import { User } from '../users/user.entity';
import { SellersController } from './sellers.controller';

@Module({
  imports: [TypeOrmModule.forFeature([Seller, User])],
  controllers: [SellersController],
  exports: [TypeOrmModule],
})
export class SellersModule {}
