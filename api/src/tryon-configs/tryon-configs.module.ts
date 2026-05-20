import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { TryonConfig } from './tryon-config.entity';
import { TryonConfigsController } from './tryon-configs.controller';

@Module({
  imports: [TypeOrmModule.forFeature([TryonConfig])],
  controllers: [TryonConfigsController],
})
export class TryonConfigsModule {}
