import { INestApplicationContext, Logger } from '@nestjs/common';
import { DataSource } from 'typeorm';
import * as bcrypt from 'bcryptjs';
import { User } from '../users/user.entity';
import { Category } from '../catalog/category.entity';
import { Product } from '../catalog/product.entity';
import { ProductImage } from '../catalog/product-image.entity';
import { ProductVariant } from '../catalog/product-variant.entity';
import { UserRole } from '../common/enums';

export async function autoSeed(app: INestApplicationContext) {
  const log = new Logger('AutoSeed');
  const ds = app.get(DataSource);

  const userRepo = ds.getRepository(User);
  const catRepo = ds.getRepository(Category);
  const prodRepo = ds.getRepository(Product);
  const imgRepo = ds.getRepository(ProductImage);
  const varRepo = ds.getRepository(ProductVariant);

  // Admin
  let admin = await userRepo.findOne({ where: { email: 'admin@shop.dev' } });
  if (!admin) {
    admin = await userRepo.save(
      userRepo.create({
        email: 'admin@shop.dev',
        passwordHash: await bcrypt.hash('admin123', 10),
        fullName: 'Admin',
        role: UserRole.ADMIN,
      }),
    );
    log.log('Created admin: admin@shop.dev / admin123');
  }

  // Categories
  const catData = [
    { name: 'Áo', slug: 'ao' },
    { name: 'Quần', slug: 'quan' },
    { name: 'Đầm', slug: 'dam' },
    { name: 'Phụ kiện', slug: 'phu-kien' },
  ];
  const cats: Record<string, Category> = {};
  for (const c of catData) {
    let cat = await catRepo.findOne({ where: { slug: c.slug } });
    if (!cat) cat = await catRepo.save(catRepo.create(c));
    cats[c.slug] = cat;
  }

  // Skip product seed nếu đã có product nào
  if ((await prodRepo.count()) > 0) {
    log.log('Products already exist, skip product seed.');
    return;
  }

  const products = [
    {
      name: 'Áo thun cotton trắng', slug: 'ao-thun-cotton-trang',
      price: 250000, originalPrice: 320000, category: cats['ao'], badge: 'Mới',
      images: ['https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=600'],
      variants: [
        { color: 'Trắng', size: 'M', sku: 'TS-WH-M', stock: 20 },
        { color: 'Trắng', size: 'L', sku: 'TS-WH-L', stock: 15 },
      ],
    },
    {
      name: 'Quần jean ống đứng', slug: 'quan-jean-ong-dung',
      price: 480000, category: cats['quan'], badge: 'Hot',
      images: ['https://images.unsplash.com/photo-1542272604-787c3835535d?w=600'],
      variants: [
        { color: 'Xanh', size: '29', sku: 'JN-29', stock: 10 },
        { color: 'Xanh', size: '30', sku: 'JN-30', stock: 12 },
      ],
    },
    {
      name: 'Đầm linen midi be', slug: 'dam-linen-midi-be',
      price: 690000, category: cats['dam'],
      images: ['https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=600'],
      variants: [
        { color: 'Be', size: 'S', sku: 'DR-BE-S', stock: 8 },
        { color: 'Be', size: 'M', sku: 'DR-BE-M', stock: 6 },
      ],
    },
  ];

  for (const p of products) {
    const prod = await prodRepo.save(
      prodRepo.create({
        name: p.name,
        slug: p.slug,
        price: String(p.price),
        originalPrice: p.originalPrice ? String(p.originalPrice) : null,
        category: p.category,
        badge: p.badge,
      }),
    );
    await imgRepo.save(
      p.images.map((url, i) => imgRepo.create({ product: prod, url, position: i })),
    );
    await varRepo.save(
      p.variants.map((v) => varRepo.create({ ...v, product: prod, priceDelta: '0' })),
    );
    log.log(`Seeded product: ${p.name}`);
  }
}
