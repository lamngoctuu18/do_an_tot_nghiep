import {
  Body,
  Controller,
  Get,
  Param,
  ParseIntPipe,
  Put,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { IsBoolean, IsInt, IsOptional, IsString, Min } from 'class-validator';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { UserRole } from '../common/enums';
import { TryonConfig } from './tryon-config.entity';

class UpdateConfigDto {
  @IsOptional() @IsString() provider?: string;
  @IsOptional() @IsString() mode?: string;
  @IsOptional() @IsBoolean() isEnabled?: boolean;
  @IsOptional() @IsInt() @Min(0) maxDailyUsage?: number;
  @IsOptional() @IsInt() @Min(1) timeoutSeconds?: number;
}

@ApiTags('Admin Try-On Configs')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard, RolesGuard)
@Roles(UserRole.ADMIN)
@Controller('admin/tryon/configs')
export class TryonConfigsController {
  constructor(@InjectRepository(TryonConfig) private repo: Repository<TryonConfig>) {}

  @Get()
  list() {
    return this.repo.find({ order: { id: 'ASC' } });
  }

  @Put(':id')
  async update(@Param('id', ParseIntPipe) id: number, @Body() dto: UpdateConfigDto) {
    await this.repo.update(id, dto);
    return this.repo.findOne({ where: { id } });
  }
}
