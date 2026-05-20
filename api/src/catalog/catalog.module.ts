import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Product } from './product.entity';
import { Category } from './category.entity';
import { ProductImage } from './product-image.entity';
import { ProductVariant } from './product-variant.entity';
import { ProductTryonAsset } from './product-tryon-asset.entity';
import { Seller } from '../sellers/seller.entity';
import { CatalogController } from './catalog.controller';

@Module({
  imports: [
    TypeOrmModule.forFeature([Product, Category, ProductImage, ProductVariant, ProductTryonAsset, Seller]),
  ],
  controllers: [CatalogController],
  exports: [TypeOrmModule],
})
export class CatalogModule {}
