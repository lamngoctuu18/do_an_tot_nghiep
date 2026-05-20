import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { TryonSession } from './tryon-session.entity';
import { TryonController } from './tryon.controller';
import { AuthModule } from '../auth/auth.module';

@Module({
  imports: [TypeOrmModule.forFeature([TryonSession]), AuthModule],
  controllers: [TryonController],
})
export class TryonModule {}
