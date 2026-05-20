import {
  Body,
  Controller,
  Delete,
  ForbiddenException,
  Get,
  NotFoundException,
  Param,
  Post,
  Put,
  Query,
  UploadedFile,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { extname, join } from 'path';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { v4 as uuid } from 'uuid';
import {
  IsArray,
  IsBoolean,
  IsEnum,
  IsNumber,
  IsOptional,
  IsString,
  ValidateNested,
} from 'class-validator';
import { Type } from 'class-transformer';
import { Product, ProductStatus } from './product.entity';
import { Category } from './category.entity';
import { ProductImage } from './product-image.entity';
import { ProductVariant } from './product-variant.entity';
import { ClothType, ProductTryonAsset } from './product-tryon-asset.entity';
import { Seller, SellerStatus } from '../sellers/seller.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { UserRole } from '../common/enums';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { ApiBearerAuth, ApiConsumes, ApiTags } from '@nestjs/swagger';

class VariantDto {
  @IsOptional() @IsString() color?: string;
  @IsOptional() @IsString() size?: string;
  @IsString() sku: string;
  @IsNumber() stock: number;
  @IsOptional() @IsNumber() priceDelta?: number;
}

class TryonAssetDto {
  @IsString() clothImageUrl: string;
  @IsOptional() @IsEnum(ClothType) clothType?: ClothType;
}

class CreateProductDto {
  @IsString() name: string;
  @IsString() slug: string;
  @IsOptional() @IsString() description?: string;
  @IsNumber() price: number;
  @IsOptional() @IsNumber() originalPrice?: number;
  @IsOptional() @IsNumber() categoryId?: number;
  @IsOptional() @IsString() badge?: string;
  @IsOptional() @IsBoolean() tryOnEnabled?: boolean;
  @IsOptional() @IsArray() images?: string[];
  @IsOptional() @ValidateNested({ each: true }) @Type(() => VariantDto) variants?: VariantDto[];
  @IsOptional() @ValidateNested({ each: true }) @Type(() => TryonAssetDto) tryonAssets?: TryonAssetDto[];
}

@ApiTags('Catalog')
@Controller()
export class CatalogController {
  constructor(
    @InjectRepository(Product) private products: Repository<Product>,
    @InjectRepository(Category) private categories: Repository<Category>,
    @InjectRepository(ProductImage) private images: Repository<ProductImage>,
    @InjectRepository(ProductVariant) private variants: Repository<ProductVariant>,
    @InjectRepository(ProductTryonAsset) private tryonAssets: Repository<ProductTryonAsset>,
    @InjectRepository(Seller) private sellers: Repository<Seller>,
  ) {}

  private async getShop(userId: number, role: string): Promise<Seller | null> {
    if (role === UserRole.ADMIN) return null;
    const shop = await this.sellers.findOne({
      where: { user: { id: userId } as any, status: SellerStatus.APPROVED },
    });
    if (!shop) throw new ForbiddenException('Shop chưa được duyệt');
    return shop;
  }

  @Get('categories')
  async listCategories() {
    return this.categories.find({ relations: ['parent'] });
  }

  @Get('products')
  async listProducts(
    @Query('q') q?: string,
    @Query('category') category?: string,
    @Query('minPrice') minPrice?: string,
    @Query('maxPrice') maxPrice?: string,
    @Query('sort') sort?: string,
    @Query('page') page = '1',
    @Query('size') size = '20',
  ) {
    const qb = this.products
      .createQueryBuilder('p')
      .leftJoinAndSelect('p.images', 'img')
      .leftJoinAndSelect('p.category', 'c')
      .where('p.status = :s', { s: ProductStatus.ACTIVE });
    if (q) qb.andWhere('p.name LIKE :q', { q: `%${q}%` });
    if (category) qb.andWhere('c.slug = :cat', { cat: category });
    if (minPrice) qb.andWhere('p.price >= :min', { min: +minPrice });
    if (maxPrice) qb.andWhere('p.price <= :max', { max: +maxPrice });
    if (sort === 'price_asc') qb.orderBy('p.price', 'ASC');
    else if (sort === 'price_desc') qb.orderBy('p.price', 'DESC');
    else if (sort === 'rating') qb.orderBy('p.ratingAvg', 'DESC');
    else qb.orderBy('p.createdAt', 'DESC');
    const take = Math.min(+size, 100);
    const skip = (Math.max(+page, 1) - 1) * take;
    qb.take(take).skip(skip);
    const [items, total] = await qb.getManyAndCount();
    return { items, total, page: +page, size: take };
  }

  @Get('products/:slug')
  async getProduct(@Param('slug') slug: string) {
    const p = await this.products.findOne({
      where: { slug },
      relations: ['images', 'variants', 'tryonAssets', 'category', 'seller'],
    });
    if (!p) throw new NotFoundException('Sản phẩm không tồn tại');
    return p;
  }

  // -------- Seller-only --------
  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.SELLER, UserRole.ADMIN)
  @Get('seller/products')
  async listMyProducts(@CurrentUser() u: any) {
    const shop = await this.getShop(u.id, u.role);
    const qb = this.products
      .createQueryBuilder('p')
      .leftJoinAndSelect('p.images', 'img')
      .leftJoinAndSelect('p.variants', 'v')
      .leftJoinAndSelect('p.category', 'c')
      .orderBy('p.createdAt', 'DESC');
    if (shop) qb.where('p.sellerId = :sid', { sid: shop.id });
    return qb.getMany();
  }

  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.SELLER, UserRole.ADMIN)
  @Get('seller/products/:id')
  async getMyProduct(@CurrentUser() u: any, @Param('id') id: number) {
    const shop = await this.getShop(u.id, u.role);
    const p = await this.products.findOne({
      where: { id },
      relations: ['images', 'variants', 'tryonAssets', 'category', 'seller'],
    });
    if (!p) throw new NotFoundException('Sản phẩm không tồn tại');
    if (shop && p.seller?.id !== shop.id) throw new ForbiddenException('Không thuộc shop của bạn');
    return p;
  }

  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.SELLER, UserRole.ADMIN)
  @Post('seller/products')
  async create(@CurrentUser() u: any, @Body() dto: CreateProductDto) {
    const shop = await this.getShop(u.id, u.role);
    const product = this.products.create({
      name: dto.name,
      slug: dto.slug,
      description: dto.description,
      price: String(dto.price),
      originalPrice: dto.originalPrice ? String(dto.originalPrice) : null,
      badge: dto.badge,
      tryOnEnabled: dto.tryOnEnabled ?? false,
      category: dto.categoryId ? ({ id: dto.categoryId } as any) : null,
      seller: shop ? ({ id: shop.id } as any) : null,
    });
    const saved = await this.products.save(product);
    if (dto.images?.length) {
      await this.images.save(
        dto.images.map((url, i) => this.images.create({ product: saved, url, position: i })),
      );
    }
    if (dto.variants?.length) {
      await this.variants.save(
        dto.variants.map((v) =>
          this.variants.create({
            product: saved,
            color: v.color,
            size: v.size,
            sku: v.sku,
            stock: v.stock,
            priceDelta: String(v.priceDelta ?? 0),
          }),
        ),
      );
    }
    if (dto.tryonAssets?.length) {
      await this.tryonAssets.save(
        dto.tryonAssets.map((t) =>
          this.tryonAssets.create({
            product: saved,
            clothImageUrl: t.clothImageUrl,
            clothType: t.clothType ?? ClothType.AUTO,
          }),
        ),
      );
    }
    return this.products.findOne({
      where: { id: saved.id },
      relations: ['images', 'variants', 'tryonAssets'],
    });
  }

  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.SELLER, UserRole.ADMIN)
  @Put('seller/products/:id')
  async update(@CurrentUser() u: any, @Param('id') id: number, @Body() dto: Partial<CreateProductDto>) {
    const shop = await this.getShop(u.id, u.role);
    const existing = await this.products.findOne({ where: { id }, relations: ['seller'] });
    if (!existing) throw new NotFoundException('Sản phẩm không tồn tại');
    if (shop && existing.seller?.id !== shop.id) throw new ForbiddenException('Không thuộc shop của bạn');
    await this.products.update(id, {
      ...(dto.name && { name: dto.name }),
      ...(dto.slug && { slug: dto.slug }),
      ...(dto.description !== undefined && { description: dto.description }),
      ...(dto.price !== undefined && { price: String(dto.price) }),
      ...(dto.originalPrice !== undefined && { originalPrice: String(dto.originalPrice) }),
      ...(dto.badge !== undefined && { badge: dto.badge }),
      ...(dto.tryOnEnabled !== undefined && { tryOnEnabled: dto.tryOnEnabled }),
      ...(dto.categoryId !== undefined && { category: { id: dto.categoryId } as any }),
    });
    if (dto.images) {
      await this.images.delete({ product: { id } as any });
      if (dto.images.length) {
        await this.images.save(
          dto.images.map((url, i) => this.images.create({ product: { id } as any, url, position: i })),
        );
      }
    }
    if (dto.variants) {
      await this.variants.delete({ product: { id } as any });
      if (dto.variants.length) {
        await this.variants.save(
          dto.variants.map((v) =>
            this.variants.create({
              product: { id } as any,
              color: v.color,
              size: v.size,
              sku: v.sku,
              stock: v.stock,
              priceDelta: String(v.priceDelta ?? 0),
            }),
          ),
        );
      }
    }
    if (dto.tryonAssets) {
      await this.tryonAssets.delete({ product: { id } as any });
      if (dto.tryonAssets.length) {
        await this.tryonAssets.save(
          dto.tryonAssets.map((t) =>
            this.tryonAssets.create({
              product: { id } as any,
              clothImageUrl: t.clothImageUrl,
              clothType: t.clothType ?? ClothType.AUTO,
            }),
          ),
        );
      }
    }
    return this.products.findOne({ where: { id }, relations: ['images', 'variants', 'tryonAssets', 'category'] });
  }

  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.SELLER, UserRole.ADMIN)
  @Delete('seller/products/:id')
  async remove(@CurrentUser() u: any, @Param('id') id: number) {
    const shop = await this.getShop(u.id, u.role);
    const existing = await this.products.findOne({ where: { id }, relations: ['seller'] });
    if (!existing) throw new NotFoundException('Sản phẩm không tồn tại');
    if (shop && existing.seller?.id !== shop.id) throw new ForbiddenException('Không thuộc shop của bạn');
    await this.products.delete(id);
    return { ok: true };
  }

  @ApiBearerAuth('JWT')
  @ApiConsumes('multipart/form-data')
  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(UserRole.SELLER, UserRole.ADMIN)
  @Post('seller/upload')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: join(process.cwd(), 'uploads'),
        filename: (_req, file, cb) =>
          cb(null, uuid().replace(/-/g, '') + extname(file.originalname || '.bin')),
      }),
      limits: { fileSize: 15 * 1024 * 1024 },
    }),
  )
  upload(@UploadedFile() file: any) {
    return { url: `/uploads/${file.filename}`, name: file.originalname };
  }
}
