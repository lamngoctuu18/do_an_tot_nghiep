import {
  BadRequestException,
  Body,
  Controller,
  ForbiddenException,
  Get,
  NotFoundException,
  Param,
  Patch,
  Post,
  UseGuards,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { DataSource, Repository } from 'typeorm';
import { IsEnum, IsInt, IsOptional, IsString, MinLength } from 'class-validator';
import { Order, OrderStatus, PaymentMethod } from './order.entity';
import { OrderItem } from './order-item.entity';
import { OrderStatusHistory } from './order-status-history.entity';
import { Cart } from '../cart/cart.entity';
import { Address } from '../users/address.entity';
import { ProductVariant } from '../catalog/product-variant.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { NotificationsService } from '../notifications/notifications.service';
import { NotificationType } from '../common/enums';

class CheckoutDto {
  @IsInt() addressId: number;
  @IsEnum(PaymentMethod) paymentMethod: PaymentMethod;
  @IsOptional() @IsString() note?: string;
}

class CancelOrderDto {
  @IsString() @MinLength(3) reason: string;
}

const FREE_SHIP_THRESHOLD = 500_000;
const SHIPPING_FEE = 30_000;

@ApiTags('Orders')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard)
@Controller('orders')
export class OrdersController {
  constructor(
    @InjectRepository(Order) private orders: Repository<Order>,
    @InjectRepository(OrderItem) private items: Repository<OrderItem>,
    @InjectRepository(Cart) private carts: Repository<Cart>,
    @InjectRepository(Address) private addresses: Repository<Address>,
    @InjectRepository(ProductVariant) private variants: Repository<ProductVariant>,
    @InjectRepository(OrderStatusHistory) private history: Repository<OrderStatusHistory>,
    private notifications: NotificationsService,
    private ds: DataSource,
  ) {}

  @Patch(':code/cancel')
  async cancel(
    @CurrentUser() u: any,
    @Param('code') code: string,
    @Body() dto: CancelOrderDto,
  ) {
    const order = await this.orders.findOne({
      where: { code },
      relations: ['user', 'items', 'items.variant', 'shop', 'shop.user'],
    });
    if (!order) throw new NotFoundException('Đơn hàng không tồn tại');
    if (!order.user || order.user.id !== u.id) throw new ForbiddenException();
    if (![OrderStatus.PENDING, OrderStatus.CONFIRMED].includes(order.status)) {
      throw new BadRequestException('Không thể hủy đơn ở trạng thái hiện tại');
    }
    return this.ds.transaction(async (em) => {
      const old = order.status;
      order.status = OrderStatus.CANCELLED;
      order.cancelReason = dto.reason;
      await em.save(order);
      for (const it of order.items) {
        if (it.variant) {
          await em.increment(ProductVariant, { id: it.variant.id }, 'stock', it.quantity);
        }
      }
      await em.save(
        em.create(OrderStatusHistory, {
          order,
          oldStatus: old,
          newStatus: OrderStatus.CANCELLED,
          changedBy: { id: u.id } as any,
          note: dto.reason,
        }),
      );
      if (order.shop?.user) {
        await this.notifications.create(
          order.shop.user.id,
          `Đơn hàng ${order.code} bị hủy`,
          `Khách hàng đã hủy đơn: ${dto.reason}`,
          NotificationType.ORDER,
          'ORDER',
          order.id,
        );
      }
      return em.findOne(Order, { where: { id: order.id }, relations: ['items'] });
    });
  }

  @Get()
  list(@CurrentUser() u: any) {
    return this.orders.find({
      where: { user: { id: u.id } },
      relations: ['items'],
      order: { placedAt: 'DESC' },
    });
  }

  @Get(':code')
  async detail(@CurrentUser() u: any, @Param('code') code: string) {
    const order = await this.orders.findOne({
      where: { code, user: { id: u.id } },
      relations: ['items', 'items.product'],
    });
    if (!order) throw new NotFoundException();
    return order;
  }

  @Post('checkout')
  async checkout(@CurrentUser() u: any, @Body() dto: CheckoutDto) {
    const address = await this.addresses.findOne({
      where: { id: dto.addressId, user: { id: u.id } },
    });
    if (!address) throw new BadRequestException('Địa chỉ không hợp lệ');

    const cart = await this.carts.findOne({
      where: { user: { id: u.id } },
      relations: ['items', 'items.variant', 'items.variant.product'],
    });
    if (!cart || cart.items.length === 0) throw new BadRequestException('Giỏ hàng trống');

    return this.ds.transaction(async (em) => {
      let subtotal = 0;
      let shopId: number | null = null;
      const orderItemsData: Partial<OrderItem>[] = [];
      for (const ci of cart.items) {
        const v = await em.findOne(ProductVariant, {
          where: { id: ci.variant.id },
          relations: ['product', 'product.seller'],
        });
        if (!v) throw new BadRequestException('Sản phẩm không tồn tại');
        if (v.stock < ci.quantity) {
          throw new BadRequestException(`Hết hàng: ${v.product.name}`);
        }
        if (v.product.seller && shopId == null) shopId = v.product.seller.id;
        const unitPrice = +v.product.price + +v.priceDelta;
        subtotal += unitPrice * ci.quantity;
        v.stock -= ci.quantity;
        await em.save(v);
        orderItemsData.push({
          product: v.product,
          variant: v,
          nameSnapshot: `${v.product.name}${v.color ? ` / ${v.color}` : ''}${v.size ? ` / ${v.size}` : ''}`,
          priceSnapshot: String(unitPrice),
          quantity: ci.quantity,
        });
      }

      const shippingFee = subtotal >= FREE_SHIP_THRESHOLD ? 0 : SHIPPING_FEE;
      const total = subtotal + shippingFee;
      const code = 'ORD' + Date.now().toString(36).toUpperCase();
      const order = em.create(Order, {
        user: { id: u.id } as any,
        shop: shopId ? ({ id: shopId } as any) : null,
        code,
        subtotal: String(subtotal),
        shippingFee: String(shippingFee),
        total: String(total),
        paymentMethod: dto.paymentMethod,
        status: OrderStatus.PENDING,
        addressSnapshot: { ...address },
        note: dto.note,
      });
      await em.save(order);
      for (const data of orderItemsData) {
        await em.save(em.create(OrderItem, { ...data, order }));
      }
      // clear cart
      await em.delete('cart_items', { cart: { id: cart.id } });

      if (dto.paymentMethod === PaymentMethod.VNPAY) {
        return {
          order: await em.findOne(Order, { where: { id: order.id }, relations: ['items'] }),
          paymentUrl: `/api/payments/vnpay/redirect?code=${code}`,
        };
      }
      return em.findOne(Order, { where: { id: order.id }, relations: ['items'] });
    });
  }
}
