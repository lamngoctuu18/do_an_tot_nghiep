import {
  BadRequestException,
  Body,
  Controller,
  Delete,
  Get,
  NotFoundException,
  Param,
  ParseIntPipe,
  Patch,
  Post,
  Put,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { InjectRepository } from '@nestjs/typeorm';
import { DataSource, Repository } from 'typeorm';
import { IsBoolean, IsOptional, IsString, MinLength } from 'class-validator';
import { Address } from '../users/address.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';

class AddressDto {
  @IsString() @MinLength(2) recipient: string;
  @IsString() @MinLength(8) phone: string;
  @IsString() @MinLength(3) line1: string;
  @IsString() ward: string;
  @IsString() district: string;
  @IsString() city: string;
  @IsOptional() @IsBoolean() isDefault?: boolean;
}

class PartialAddressDto {
  @IsOptional() @IsString() recipient?: string;
  @IsOptional() @IsString() phone?: string;
  @IsOptional() @IsString() line1?: string;
  @IsOptional() @IsString() ward?: string;
  @IsOptional() @IsString() district?: string;
  @IsOptional() @IsString() city?: string;
  @IsOptional() @IsBoolean() isDefault?: boolean;
}

@ApiTags('Addresses')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard)
@Controller('users/me/addresses')
export class AddressesController {
  constructor(
    @InjectRepository(Address) private repo: Repository<Address>,
    private ds: DataSource,
  ) {}

  @Get()
  list(@CurrentUser() u: any) {
    return this.repo.find({
      where: { user: { id: u.id } },
      order: { isDefault: 'DESC', id: 'DESC' },
    });
  }

  @Post()
  async create(@CurrentUser() u: any, @Body() dto: AddressDto) {
    return this.ds.transaction(async (em) => {
      if (dto.isDefault) {
        await em.update(Address, { user: { id: u.id } as any }, { isDefault: false });
      }
      const a = em.create(Address, { ...dto, user: { id: u.id } as any });
      return em.save(a);
    });
  }

  @Put(':id')
  async update(
    @CurrentUser() u: any,
    @Param('id', ParseIntPipe) id: number,
    @Body() dto: PartialAddressDto,
  ) {
    const a = await this.repo.findOne({ where: { id, user: { id: u.id } as any } });
    if (!a) throw new NotFoundException('Địa chỉ không tồn tại');
    return this.ds.transaction(async (em) => {
      if (dto.isDefault) {
        await em.update(Address, { user: { id: u.id } as any }, { isDefault: false });
      }
      await em.update(Address, id, dto);
      return em.findOne(Address, { where: { id } });
    });
  }

  @Patch(':id/default')
  async setDefault(@CurrentUser() u: any, @Param('id', ParseIntPipe) id: number) {
    const a = await this.repo.findOne({ where: { id, user: { id: u.id } as any } });
    if (!a) throw new NotFoundException('Địa chỉ không tồn tại');
    return this.ds.transaction(async (em) => {
      await em.update(Address, { user: { id: u.id } as any }, { isDefault: false });
      await em.update(Address, id, { isDefault: true });
      return em.findOne(Address, { where: { id } });
    });
  }

  @Delete(':id')
  async remove(@CurrentUser() u: any, @Param('id', ParseIntPipe) id: number) {
    const a = await this.repo.findOne({ where: { id, user: { id: u.id } as any } });
    if (!a) throw new NotFoundException('Địa chỉ không tồn tại');
    await this.repo.delete(id);
    return { success: true };
  }
}
