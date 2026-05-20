import { Body, Controller, Get, Post, UseGuards } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { IsOptional, IsString } from 'class-validator';
import { Seller, SellerStatus } from './seller.entity';
import { User } from '../users/user.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';

class ApplySellerDto {
  @IsString() shopName: string;
  @IsString() slug: string;
  @IsOptional() @IsString() description?: string;
  @IsOptional() @IsString() logoUrl?: string;
  @IsOptional() @IsString() bannerUrl?: string;
}

@ApiTags('Sellers')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard)
@Controller('sellers')
export class SellersController {
  constructor(
    @InjectRepository(Seller) private sellers: Repository<Seller>,
    @InjectRepository(User) private users: Repository<User>,
  ) {}

  @Post('apply')
  async apply(@CurrentUser() u: any, @Body() dto: ApplySellerDto) {
    const exist = await this.sellers.findOne({ where: { user: { id: u.id } } });
    if (exist) return exist;
    return this.sellers.save(
      this.sellers.create({
        user: { id: u.id } as any,
        shopName: dto.shopName,
        slug: dto.slug,
        description: dto.description,
        logoUrl: dto.logoUrl,
        bannerUrl: dto.bannerUrl,
        status: SellerStatus.PENDING,
      }),
    );
  }

  @Get('me')
  me(@CurrentUser() u: any) {
    return this.sellers.findOne({ where: { user: { id: u.id } } });
  }
}
