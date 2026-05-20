import {
  Controller,
  Delete,
  Get,
  NotFoundException,
  Param,
  Post,
  UseGuards,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Wishlist } from './wishlist.entity';
import { Product } from '../catalog/product.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';

@ApiTags('Wishlist')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard)
@Controller('wishlist')
export class WishlistController {
  constructor(
    @InjectRepository(Wishlist) private wl: Repository<Wishlist>,
    @InjectRepository(Product) private products: Repository<Product>,
  ) {}

  @Get()
  list(@CurrentUser() u: any) {
    return this.wl.find({ where: { user: { id: u.id } } });
  }

  @Post(':productId')
  async add(@CurrentUser() u: any, @Param('productId') productId: number) {
    const product = await this.products.findOne({ where: { id: productId } });
    if (!product) throw new NotFoundException();
    const exists = await this.wl.findOne({
      where: { user: { id: u.id }, product: { id: productId } },
    });
    if (exists) return exists;
    return this.wl.save(this.wl.create({ user: { id: u.id } as any, product }));
  }

  @Delete(':productId')
  async remove(@CurrentUser() u: any, @Param('productId') productId: number) {
    await this.wl.delete({ user: { id: u.id } as any, product: { id: productId } as any });
    return { ok: true };
  }
}
