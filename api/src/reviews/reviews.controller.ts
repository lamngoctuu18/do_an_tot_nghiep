import { Body, Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { IsArray, IsInt, IsOptional, IsString, Max, Min } from 'class-validator';
import { Review } from './review.entity';
import { Product } from '../catalog/product.entity';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/current-user.decorator';
import { ApiBearerAuth, ApiTags } from '@nestjs/swagger';

class CreateReviewDto {
  @IsInt() @Min(1) @Max(5) rating: number;
  @IsOptional() @IsString() comment?: string;
  @IsOptional() @IsArray() images?: string[];
}

@ApiTags('Reviews')
@Controller('products/:productId/reviews')
export class ReviewsController {
  constructor(
    @InjectRepository(Review) private reviews: Repository<Review>,
    @InjectRepository(Product) private products: Repository<Product>,
  ) {}

  @Get()
  list(@Param('productId') productId: number) {
    return this.reviews.find({
      where: { product: { id: productId } },
      relations: ['user'],
      order: { createdAt: 'DESC' },
    });
  }

  @ApiBearerAuth('JWT')
  @UseGuards(JwtAuthGuard)
  @Post()
  async create(
    @CurrentUser() u: any,
    @Param('productId') productId: number,
    @Body() dto: CreateReviewDto,
  ) {
    const review = await this.reviews.save(
      this.reviews.create({
        product: { id: productId } as any,
        user: { id: u.id } as any,
        rating: dto.rating,
        comment: dto.comment,
        images: dto.images,
      }),
    );
    const stats = await this.reviews
      .createQueryBuilder('r')
      .select('AVG(r.rating)', 'avg')
      .addSelect('COUNT(r.id)', 'cnt')
      .where('r.productId = :pid', { pid: productId })
      .getRawOne();
    await this.products.update(productId, {
      ratingAvg: String(parseFloat(stats.avg).toFixed(2)),
      ratingCount: +stats.cnt,
    });
    return review;
  }
}
