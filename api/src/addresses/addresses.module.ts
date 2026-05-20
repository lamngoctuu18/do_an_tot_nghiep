import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Address } from '../users/address.entity';
import { AddressesController } from './addresses.controller';

@Module({
  imports: [TypeOrmModule.forFeature([Address])],
  controllers: [AddressesController],
})
export class AddressesModule {}
