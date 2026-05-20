import {
  BadRequestException,
  Body,
  Controller,
  ForbiddenException,
  Get,
  NotFoundException,
  Param,
  Patch,
  Query,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { InjectRepository } from '@nestjs/typeorm';
import { DataSource, MoreThanOrEqual, Repository } from 'typeorm';
import { IsEnum, IsOptional } from 'class-validator';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { NotificationType, UserRole } from '../common/enums';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { Order, OrderStatus } from '../orders/order.entity';
import { OrderStatusHistory } from '../orders/order-status-history.entity';
import { Seller, SellerStatus } from '../sellers/seller.entity';
import { Product, ProductStatus } from '../catalog/product.entity';
import { ProductVariant } from '../catalog/product-variant.entity';
import { TryonSession } from '../tryon/tryon-session.entity';
import { NotificationsService } from '../notifications/notifications.service';

class UpdateOrderStatusDto {
  @IsEnum(OrderStatus) orderStatus: OrderStatus;
}

const SELLER_TRANSITIONS: Record<string, OrderStatus[]> = {
  [OrderStatus.CONFIRMED]: [OrderStatus.PACKING, OrderStatus.CANCELLED],
  [OrderStatus.PACKING]: [OrderStatus.SHIPPING],
  [OrderStatus.SHIPPING]: [OrderStatus.COMPLETED],
};

@ApiTags('Seller')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard, RolesGuard)
@Roles(UserRole.SELLER, UserRole.ADMIN)
@Controller('seller')
export class SellerOrdersController {
  constructor(
    @InjectRepository(Order) private orders: Repository<Order>,
    @InjectRepository(OrderStatusHistory) private history: Repository<OrderStatusHistory>,
    @InjectRepository(Seller) private sellers: Repository<Seller>,
    @InjectRepository(Product) private products: Repository<Product>,
    @InjectRepository(ProductVariant) private variants: Repository<ProductVariant>,
    @InjectRepository(TryonSession) private tryons: Repository<TryonSession>,
    private notifications: NotificationsService,
    private ds: DataSource,
  ) {}

  private async getShop(userId: number): Promise<Seller> {
    const shop = await this.sellers.findOne({
      where: { user: { id: userId } as any, status: SellerStatus.APPROVED },
    });
    if (!shop) throw new ForbiddenException('Shop chưa được duyệt');
    return shop;
  }

  private ordersQuery(shopId: number) {
    return this.orders
      .createQueryBuilder('o')
      .innerJoin('o.items', 'oi')
      .innerJoin('oi.product', 'p')
      .innerJoin('p.seller', 's')
      .leftJoinAndSelect('o.items', 'i')
      .leftJoinAndSelect('o.user', 'u')
      .where('s.id = :shopId', { shopId })
      .distinct(true);
  }

  // -------- Dashboard --------
  @Get('dashboard/summary')
  async dashboard(@CurrentUser() u: any) {
    const shop = await this.getShop(u.id);
    const now = new Date();
    const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const baseQb = () => this.ordersQuery(shop.id);

    const [pending, confirmed, shipping, completed] = await Promise.all([
      baseQb().andWhere('o.status = :s', { s: OrderStatus.PENDING }).getCount(),
      baseQb().andWhere('o.status = :s', { s: OrderStatus.CONFIRMED }).getCount(),
      baseQb().andWhere('o.status = :s', { s: OrderStatus.SHIPPING }).getCount(),
      baseQb().andWhere('o.status = :s', { s: OrderStatus.COMPLETED }).getCount(),
    ]);
    const todayRow = await baseQb()
      .andWhere('o.status = :s AND o.placedAt >= :d', {
        s: OrderStatus.COMPLETED,
        d: startToday,
      })
      .select('COALESCE(SUM(o.total),0)', 'sum')
      .getRawOne();
    const monthRow = await baseQb()
      .andWhere('o.status = :s AND o.placedAt >= :d', {
        s: OrderStatus.COMPLETED,
        d: startMonth,
      })
      .select('COALESCE(SUM(o.total),0)', 'sum')
      .getRawOne();
    const lowStock = await this.variants
      .createQueryBuilder('v')
      .innerJoin('v.product', 'p')
      .innerJoin('p.seller', 's')
      .where('s.id = :shopId AND v.stock <= 5', { shopId: shop.id })
      .getCount();
    return {
      pendingOrders: pending,
      confirmedOrders: confirmed,
      shippingOrders: shipping,
      completedOrders: completed,
      todayRevenue: +(todayRow?.sum ?? 0),
      monthRevenue: +(monthRow?.sum ?? 0),
      lowStockVariants: lowStock,
    };
  }

  // -------- Try-on statistics --------
  @Get('tryon/statistics')
  async tryonStats(@CurrentUser() u: any) {
    const shop = await this.getShop(u.id);
    const rows = await this.tryons
      .createQueryBuilder('t')
      .innerJoin('t.product', 'p')
      .innerJoin('p.seller', 's')
      .select('p.id', 'productId')
      .addSelect('p.name', 'productName')
      .addSelect('COUNT(*)', 'tryonCount')
      .where('s.id = :shopId', { shopId: shop.id })
      .groupBy('p.id')
      .orderBy('tryonCount', 'DESC')
      .limit(50)
      .getRawMany();
    return rows.map((r) => ({
      productId: +r.productId,
      productName: r.productName,
      tryonCount: +r.tryonCount,
    }));
  }

  // -------- Orders --------
  @Get('orders')
  async list(
    @CurrentUser() u: any,
    @Query('status') status?: OrderStatus,
    @Query('page') page = '1',
    @Query('limit') limit = '20',
  ) {
    const shop = await this.getShop(u.id);
    const qb = this.ordersQuery(shop.id).orderBy('o.placedAt', 'DESC');
    if (status) qb.andWhere('o.status = :status', { status });
    const take = Math.min(+limit, 100);
    qb.take(take).skip((Math.max(+page, 1) - 1) * take);
    const [items, total] = await qb.getManyAndCount();
    return { items, total, page: +page, limit: take };
  }

  @Get('orders/:code')
  async detail(@CurrentUser() u: any, @Param('code') code: string) {
    const shop = await this.getShop(u.id);
    const order = await this.ordersQuery(shop.id)
      .andWhere('o.code = :code', { code })
      .leftJoinAndSelect('i.product', 'iproduct')
      .leftJoinAndSelect('i.variant', 'ivariant')
      .getOne();
    if (!order) throw new NotFoundException('Đơn hàng không tồn tại');
    return order;
  }

  @Patch('orders/:code/confirm')
  async confirm(@CurrentUser() u: any, @Param('code') code: string) {
    const shop = await this.getShop(u.id);
    const order = await this.ordersQuery(shop.id)
      .andWhere('o.code = :code', { code })
      .leftJoinAndSelect('o.user', 'ouser')
      .getOne();
    if (!order) throw new NotFoundException('Đơn hàng không tồn tại');
    if (order.status !== OrderStatus.PENDING) {
      throw new BadRequestException('Đơn hàng không ở trạng thái chờ xác nhận');
    }
    return this.ds.transaction(async (em) => {
      const old = order.status;
      order.status = OrderStatus.CONFIRMED;
      await em.save(order);
      await em.save(
        em.create(OrderStatusHistory, {
          order,
          oldStatus: old,
          newStatus: OrderStatus.CONFIRMED,
          changedBy: { id: u.id } as any,
        }),
      );
      if (order.user) {
        await this.notifications.create(
          order.user.id,
          `Đơn ${order.code} đã được xác nhận`,
          'Shop đã xác nhận đơn hàng của bạn.',
          NotificationType.ORDER,
          'ORDER',
          order.id,
        );
      }
      return order;
    });
  }

  @Patch('orders/:code/status')
  async updateStatus(
    @CurrentUser() u: any,
    @Param('code') code: string,
    @Body() dto: UpdateOrderStatusDto,
  ) {
    const shop = await this.getShop(u.id);
    const order = await this.ordersQuery(shop.id)
      .andWhere('o.code = :code', { code })
      .leftJoinAndSelect('o.user', 'ouser')
      .getOne();
    if (!order) throw new NotFoundException('Đơn hàng không tồn tại');
    const allowed = SELLER_TRANSITIONS[order.status] ?? [];
    if (!allowed.includes(dto.orderStatus)) {
      throw new BadRequestException(
        `Không thể chuyển trạng thái từ ${order.status} sang ${dto.orderStatus}`,
      );
    }
    return this.ds.transaction(async (em) => {
      const old = order.status;
      order.status = dto.orderStatus;
      await em.save(order);
      await em.save(
        em.create(OrderStatusHistory, {
          order,
          oldStatus: old,
          newStatus: dto.orderStatus,
          changedBy: { id: u.id } as any,
        }),
      );
      if (order.user) {
        await this.notifications.create(
          order.user.id,
          `Đơn ${order.code} cập nhật trạng thái`,
          `Đơn của bạn đã chuyển sang ${dto.orderStatus}.`,
          NotificationType.ORDER,
          'ORDER',
          order.id,
        );
      }
      return order;
    });
  }

  // -------- Submit product for review --------
  @Patch('products/:id/submit')
  async submitProduct(@CurrentUser() u: any, @Param('id') id: number) {
    const shop = await this.getShop(u.id);
    const product = await this.products.findOne({
      where: { id, seller: { id: shop.id } as any },
      relations: ['images', 'variants', 'tryonAssets', 'category'],
    });
    if (!product) throw new NotFoundException('Sản phẩm không tồn tại');
    if (product.status === ProductStatus.PENDING) {
      throw new BadRequestException('Sản phẩm đang chờ duyệt');
    }
    if (product.status === ProductStatus.ACTIVE) {
      throw new BadRequestException('Sản phẩm đã được duyệt');
    }
    if (!product.name?.trim() || product.name.trim().length < 2) {
      throw new BadRequestException('Tên sản phẩm chưa hợp lệ');
    }
    if (!product.category) {
      throw new BadRequestException('Chưa chọn danh mục');
    }
    if (!product.price || +product.price <= 0) {
      throw new BadRequestException('Giá sản phẩm phải lớn hơn 0');
    }
    const validImages = (product.images ?? []).filter((i) => i.url?.trim());
    if (validImages.length === 0) {
      throw new BadRequestException('Cần ít nhất một ảnh sản phẩm có URL hợp lệ');
    }
    const validVariants = (product.variants ?? []).filter(
      (v) => v.stock >= 0 && (v.color?.trim() || v.size?.trim()),
    );
    if (validVariants.length === 0) {
      throw new BadRequestException(
        'Cần ít nhất một biến thể hợp lệ (có màu hoặc size và tồn kho ≥ 0)',
      );
    }
    if (product.tryOnEnabled) {
      const validAssets = (product.tryonAssets ?? []).filter((a) => a.clothImageUrl?.trim());
      if (validAssets.length === 0) {
        throw new BadRequestException(
          'Bật thử đồ ảo cần ít nhất 1 ảnh CLOTH hợp lệ (tryon asset)',
        );
      }
    }
    product.status = ProductStatus.PENDING;
    product.rejectReason = null;
    return this.products.save(product);
  }
}
