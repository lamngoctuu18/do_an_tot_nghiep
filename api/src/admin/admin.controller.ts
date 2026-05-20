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
  Query,
  UseGuards,
} from '@nestjs/common';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';
import { InjectRepository } from '@nestjs/typeorm';
import { MoreThanOrEqual, Repository } from 'typeorm';
import { IsInt, IsOptional, IsString, Matches, MinLength } from 'class-validator';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { NotificationType, UserRole, UserStatus } from '../common/enums';
import { User } from '../users/user.entity';
import { Seller, SellerStatus } from '../sellers/seller.entity';
import { Order, OrderStatus } from '../orders/order.entity';
import { Product, ProductStatus } from '../catalog/product.entity';
import { Category } from '../catalog/category.entity';
import { Review } from '../reviews/review.entity';
import { ReviewStatus } from '../common/enums';
import { TryonSession } from '../tryon/tryon-session.entity';
import { NotificationsService } from '../notifications/notifications.service';

class ReasonDto {
  @IsString() @MinLength(3) reason: string;
}

class CategoryDto {
  @IsString() @MinLength(2) name: string;
  @IsString() @MinLength(2) @Matches(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, {
    message: 'slug chỉ chứa chữ thường, số và dấu gạch ngang',
  })
  slug: string;
  @IsOptional() @IsInt() parentId?: number;
}

class PartialCategoryDto {
  @IsOptional() @IsString() @MinLength(2) name?: string;
  @IsOptional() @IsString() @MinLength(2) @Matches(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, {
    message: 'slug chỉ chứa chữ thường, số và dấu gạch ngang',
  })
  slug?: string;
  @IsOptional() @IsInt() parentId?: number;
}

@ApiTags('Admin')
@ApiBearerAuth('JWT')
@UseGuards(JwtAuthGuard, RolesGuard)
@Roles(UserRole.ADMIN)
@Controller('admin')
export class AdminController {
  constructor(
    @InjectRepository(User) private users: Repository<User>,
    @InjectRepository(Seller) private sellers: Repository<Seller>,
    @InjectRepository(Order) private orders: Repository<Order>,
    @InjectRepository(Product) private products: Repository<Product>,
    @InjectRepository(Category) private categories: Repository<Category>,
    @InjectRepository(Review) private reviews: Repository<Review>,
    @InjectRepository(TryonSession) private tryons: Repository<TryonSession>,
    private notifications: NotificationsService,
  ) {}

  // -------- Dashboard --------
  @Get(['stats', 'dashboard/summary'])
  async stats() {
    const now = new Date();
    const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startMonth = new Date(now.getFullYear(), now.getMonth(), 1);
    const [usersCount, sellersCount, productsCount, ordersCount] = await Promise.all([
      this.users.count(),
      this.sellers.count(),
      this.products.count(),
      this.orders.count(),
    ]);
    const revenueRow = await this.orders
      .createQueryBuilder('o')
      .select('COALESCE(SUM(o.total),0)', 'sum')
      .where('o.status = :s', { s: OrderStatus.COMPLETED })
      .getRawOne();
    const todayRow = await this.orders
      .createQueryBuilder('o')
      .select('COALESCE(SUM(o.total),0)', 'sum')
      .where('o.status = :s AND o.placedAt >= :d', { s: OrderStatus.COMPLETED, d: startToday })
      .getRawOne();
    const monthRow = await this.orders
      .createQueryBuilder('o')
      .select('COALESCE(SUM(o.total),0)', 'sum')
      .where('o.status = :s AND o.placedAt >= :d', { s: OrderStatus.COMPLETED, d: startMonth })
      .getRawOne();
    const tryonToday = await this.tryons.count({ where: { createdAt: MoreThanOrEqual(startToday) } });
    return {
      totalUsers: usersCount,
      totalSellers: sellersCount,
      totalProducts: productsCount,
      totalOrders: ordersCount,
      revenue: +revenueRow.sum,
      todayRevenue: +todayRow.sum,
      monthRevenue: +monthRow.sum,
      tryonCountToday: tryonToday,
    };
  }

  // -------- Users --------
  @Get('users')
  listUsers(@Query('q') q?: string, @Query('role') role?: UserRole, @Query('status') status?: UserStatus) {
    const qb = this.users.createQueryBuilder('u');
    if (q) qb.where('u.email LIKE :q OR u.fullName LIKE :q', { q: `%${q}%` });
    if (role) qb.andWhere('u.role = :role', { role });
    if (status) qb.andWhere('u.status = :status', { status });
    return qb.orderBy('u.createdAt', 'DESC').take(200).getMany();
  }

  @Patch('users/:id/lock')
  async lockUser(@Param('id', ParseIntPipe) id: number) {
    await this.users.update(id, { status: UserStatus.LOCKED });
    return this.users.findOne({ where: { id } });
  }

  @Patch('users/:id/unlock')
  async unlockUser(@Param('id', ParseIntPipe) id: number) {
    await this.users.update(id, { status: UserStatus.ACTIVE });
    return this.users.findOne({ where: { id } });
  }

  // legacy alias
  @Patch('users/:id/status')
  async patchUser(@Param('id', ParseIntPipe) id: number, @Body('status') status: UserStatus) {
    await this.users.update(id, { status });
    return this.users.findOne({ where: { id } });
  }

  // -------- Sellers / Shops --------
  @Get(['sellers', 'shops'])
  listSellers(@Query('status') status?: SellerStatus) {
    const qb = this.sellers.createQueryBuilder('s').leftJoinAndSelect('s.user', 'u');
    if (status) qb.where('s.status = :status', { status });
    return qb.orderBy('s.createdAt', 'DESC').getMany();
  }

  @Patch(['shops/:id/approve', 'sellers/:id/approve'])
  async approveShop(@Param('id', ParseIntPipe) id: number) {
    const shop = await this.sellers.findOne({ where: { id }, relations: ['user'] });
    if (!shop) throw new NotFoundException('Shop không tồn tại');
    shop.status = SellerStatus.APPROVED;
    shop.rejectReason = null;
    await this.sellers.save(shop);
    if (shop.user) {
      await this.users.update(shop.user.id, { role: UserRole.SELLER });
      await this.notifications.create(
        shop.user.id,
        'Shop được phê duyệt',
        `Shop "${shop.shopName}" đã được duyệt. Bạn có thể bắt đầu bán hàng.`,
        NotificationType.SHOP,
        'SHOP',
        shop.id,
      );
    }
    return shop;
  }

  @Patch(['shops/:id/reject', 'sellers/:id/reject'])
  async rejectShop(@Param('id', ParseIntPipe) id: number, @Body() dto: ReasonDto) {
    const shop = await this.sellers.findOne({ where: { id }, relations: ['user'] });
    if (!shop) throw new NotFoundException();
    shop.status = SellerStatus.REJECTED;
    shop.rejectReason = dto.reason;
    await this.sellers.save(shop);
    if (shop.user) {
      await this.notifications.create(
        shop.user.id,
        'Shop bị từ chối',
        `Lý do: ${dto.reason}`,
        NotificationType.SHOP,
        'SHOP',
        shop.id,
      );
    }
    return shop;
  }

  @Patch('sellers/:id/status')
  async patchSeller(@Param('id', ParseIntPipe) id: number, @Body('status') status: SellerStatus) {
    await this.sellers.update(id, { status });
    return this.sellers.findOne({ where: { id }, relations: ['user'] });
  }

  // -------- Products --------
  @Get('products')
  listProducts(@Query('status') status?: ProductStatus) {
    const qb = this.products
      .createQueryBuilder('p')
      .leftJoinAndSelect('p.seller', 's')
      .leftJoinAndSelect('p.category', 'c');
    if (status) qb.where('p.status = :status', { status });
    return qb.orderBy('p.createdAt', 'DESC').take(200).getMany();
  }

  @Patch('products/:id/approve')
  async approveProduct(@Param('id', ParseIntPipe) id: number) {
    const p = await this.products.findOne({ where: { id }, relations: ['seller', 'seller.user'] });
    if (!p) throw new NotFoundException();
    p.status = ProductStatus.ACTIVE;
    p.rejectReason = null;
    await this.products.save(p);
    if (p.seller?.user) {
      await this.notifications.create(
        p.seller.user.id,
        'Sản phẩm được duyệt',
        `Sản phẩm "${p.name}" đã được duyệt và hiển thị.`,
        NotificationType.PRODUCT,
        'PRODUCT',
        p.id,
      );
    }
    return p;
  }

  @Patch('products/:id/reject')
  async rejectProduct(@Param('id', ParseIntPipe) id: number, @Body() dto: ReasonDto) {
    const p = await this.products.findOne({ where: { id }, relations: ['seller', 'seller.user'] });
    if (!p) throw new NotFoundException();
    p.status = ProductStatus.REJECTED;
    p.rejectReason = dto.reason;
    await this.products.save(p);
    if (p.seller?.user) {
      await this.notifications.create(
        p.seller.user.id,
        'Sản phẩm bị từ chối',
        `Sản phẩm "${p.name}". Lý do: ${dto.reason}`,
        NotificationType.PRODUCT,
        'PRODUCT',
        p.id,
      );
    }
    return p;
  }

  // -------- Reviews --------
  @Get('reviews')
  async listReviews(@Query('status') status?: string) {
    const qb = this.reviews
      .createQueryBuilder('r')
      .leftJoinAndSelect('r.user', 'u')
      .leftJoinAndSelect('r.product', 'p')
      .orderBy('r.createdAt', 'DESC');
    if (status) qb.where('r.status = :s', { s: status });
    return qb.getMany();
  }

  @Patch('reviews/:id/hide')
  async hideReview(@Param('id', ParseIntPipe) id: number) {
    await this.reviews.update(id, { status: ReviewStatus.HIDDEN });
    return this.reviews.findOne({ where: { id } });
  }

  @Patch('reviews/:id/unhide')
  async unhideReview(@Param('id', ParseIntPipe) id: number) {
    await this.reviews.update(id, { status: ReviewStatus.VISIBLE });
    return this.reviews.findOne({ where: { id } });
  }

  // -------- Categories --------
  @Post('categories')
  async createCategory(@Body() dto: CategoryDto) {
    const exists = await this.categories.findOne({ where: { slug: dto.slug } });
    if (exists) throw new BadRequestException('Slug đã tồn tại');
    if (dto.parentId) {
      const parent = await this.categories.findOne({ where: { id: dto.parentId } });
      if (!parent) throw new BadRequestException('Danh mục cha không tồn tại');
    }
    return this.categories.save(
      this.categories.create({
        name: dto.name,
        slug: dto.slug,
        parent: dto.parentId ? ({ id: dto.parentId } as any) : null,
      }),
    );
  }

  @Put('categories/:id')
  async updateCategory(@Param('id', ParseIntPipe) id: number, @Body() dto: PartialCategoryDto) {
    const current = await this.categories.findOne({ where: { id } });
    if (!current) throw new NotFoundException('Danh mục không tồn tại');
    if (dto.slug && dto.slug !== current.slug) {
      const dup = await this.categories.findOne({ where: { slug: dto.slug } });
      if (dup && dup.id !== id) throw new BadRequestException('Slug đã tồn tại');
    }
    if (dto.parentId !== undefined && dto.parentId !== null) {
      if (dto.parentId === id) throw new BadRequestException('Không thể đặt làm cha của chính nó');
      const parent = await this.categories.findOne({ where: { id: dto.parentId } });
      if (!parent) throw new BadRequestException('Danh mục cha không tồn tại');
    }
    const payload: any = { ...dto };
    if (dto.parentId !== undefined) payload.parent = dto.parentId ? { id: dto.parentId } : null;
    delete payload.parentId;
    await this.categories.update(id, payload);
    return this.categories.findOne({ where: { id } });
  }

  @Delete('categories/:id')
  async deleteCategory(@Param('id', ParseIntPipe) id: number) {
    const cat = await this.categories.findOne({ where: { id } });
    if (!cat) throw new NotFoundException('Danh mục không tồn tại');
    const childCount = await this.categories.count({ where: { parent: { id } as any } });
    if (childCount > 0) {
      throw new BadRequestException('Danh mục còn danh mục con, không thể xoá');
    }
    await this.categories.delete(id);
    return { success: true };
  }

  // -------- Orders (legacy) --------
  @Get('orders')
  listOrders() {
    return this.orders.find({
      relations: ['user', 'items'],
      order: { placedAt: 'DESC' },
      take: 200,
    });
  }

  @Patch('orders/:code/status')
  async patchOrder(@Param('code') code: string, @Body('status') status: OrderStatus) {
    const order = await this.orders.findOne({ where: { code } });
    if (!order) throw new NotFoundException();
    order.status = status;
    return this.orders.save(order);
  }

  // -------- Reports --------
  @Get('reports/revenue')
  async revenueReport(@Query('from_date') from?: string, @Query('to_date') to?: string) {
    const fromD = from ? new Date(from) : new Date(Date.now() - 30 * 86400000);
    const toD = to ? new Date(to) : new Date();
    const rows = await this.orders
      .createQueryBuilder('o')
      .select("DATE(o.placedAt)", 'date')
      .addSelect('COALESCE(SUM(o.total),0)', 'revenue')
      .addSelect('COUNT(*)', 'orders')
      .where('o.status = :s AND o.placedAt BETWEEN :from AND :to', {
        s: OrderStatus.COMPLETED,
        from: fromD,
        to: toD,
      })
      .groupBy('DATE(o.placedAt)')
      .orderBy('DATE(o.placedAt)', 'ASC')
      .getRawMany();
    return rows.map((r) => ({ date: r.date, revenue: +r.revenue, orders: +r.orders }));
  }

  @Get('reports/top-products')
  async topProducts(@Query('limit') limit = '10') {
    const rows = await this.orders.manager
      .createQueryBuilder()
      .select('p.id', 'productId')
      .addSelect('p.name', 'productName')
      .addSelect('SUM(oi.quantity)', 'sold')
      .from('order_items', 'oi')
      .innerJoin('orders', 'o', 'o.id = oi.orderId')
      .innerJoin('products', 'p', 'p.id = oi.productId')
      .where('o.status = :s', { s: OrderStatus.COMPLETED })
      .groupBy('p.id')
      .orderBy('sold', 'DESC')
      .limit(Math.min(+limit, 50))
      .getRawMany();
    return rows.map((r) => ({ productId: +r.productId, productName: r.productName, sold: +r.sold }));
  }

  @Get('reports/tryon')
  async tryonReport() {
    const totalRow = await this.tryons.createQueryBuilder('t').select('COUNT(*)', 'c').getRawOne();
    const okRow = await this.tryons
      .createQueryBuilder('t')
      .select('COUNT(*)', 'c')
      .where("t.status IN ('OK', 'SUCCESS')")
      .getRawOne();
    const failedRow = await this.tryons
      .createQueryBuilder('t')
      .select('COUNT(*)', 'c')
      .where("t.status IN ('FAILED', 'ERROR')")
      .getRawOne();
    const byBackend = await this.tryons
      .createQueryBuilder('t')
      .select('t.backend', 'backend')
      .addSelect('COUNT(*)', 'count')
      .groupBy('t.backend')
      .getRawMany();
    return {
      total: +totalRow.c,
      success: +okRow.c,
      failed: +failedRow.c,
      byBackend: byBackend.map((b) => ({ backend: b.backend, count: +b.count })),
    };
  }
}
