import {
  BadRequestException,
  Body,
  Controller,
  Delete,
  Get,
  NotFoundException,
  Param,
  Patch,
  Post,
  UseGuards,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { IsInt, Min } from 'class-validator';
import { Cart } from './cart.entity';
import { CartItem } from './cart-item.entity';
import { ProductVariant } from '../catalog/product-variant.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';

class AddItemDto {
  @IsInt() variantId: number;
  @IsInt() @Min(1) quantity: number;
}

class UpdateItemDto {
  @IsInt() @Min(1) quantity: number;
}

@ApiTags('Cart')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard)
@Controller('cart')
export class CartController {
  constructor(
    @InjectRepository(Cart) private carts: Repository<Cart>,
    @InjectRepository(CartItem) private items: Repository<CartItem>,
    @InjectRepository(ProductVariant) private variants: Repository<ProductVariant>,
  ) {}

  private async getOrCreate(userId: number) {
    let cart = await this.carts.findOne({
      where: { user: { id: userId } },
      relations: ['items', 'items.variant', 'items.variant.product', 'items.variant.product.images'],
    });
    if (!cart) {
      cart = await this.carts.save(this.carts.create({ user: { id: userId } as any }));
      cart.items = [];
    }
    return cart;
  }

  @Get()
  async get(@CurrentUser() u: any) {
    return this.getOrCreate(u.id);
  }

  @Post('items')
  async add(@CurrentUser() u: any, @Body() dto: AddItemDto) {
    const variant = await this.variants.findOne({ where: { id: dto.variantId } });
    if (!variant) throw new NotFoundException('Variant không tồn tại');
    if (variant.stock < dto.quantity) throw new BadRequestException('Không đủ tồn kho');
    const cart = await this.getOrCreate(u.id);
    const existing = cart.items.find((i) => i.variant.id === variant.id);
    if (existing) {
      existing.quantity += dto.quantity;
      await this.items.save(existing);
    } else {
      await this.items.save(this.items.create({ cart, variant, quantity: dto.quantity }));
    }
    return this.getOrCreate(u.id);
  }

  @Patch('items/:id')
  async patch(@CurrentUser() u: any, @Param('id') id: number, @Body() dto: UpdateItemDto) {
    const item = await this.items.findOne({ where: { id }, relations: ['cart', 'cart.user', 'variant'] });
    if (!item || item.cart.user.id !== u.id) throw new NotFoundException();
    item.quantity = dto.quantity;
    await this.items.save(item);
    return this.getOrCreate(u.id);
  }

  @Delete('items/:id')
  async remove(@CurrentUser() u: any, @Param('id') id: number) {
    const item = await this.items.findOne({ where: { id }, relations: ['cart', 'cart.user'] });
    if (!item || item.cart.user.id !== u.id) throw new NotFoundException();
    await this.items.delete(id);
    return this.getOrCreate(u.id);
  }
}
