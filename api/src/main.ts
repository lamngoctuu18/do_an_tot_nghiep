import 'reflect-metadata';
import { config as loadEnv } from 'dotenv';
loadEnv();

import { NestFactory } from '@nestjs/core';
import { ValidationPipe, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { join } from 'path';
import { mkdirSync } from 'fs';
import { ensureDatabase } from './bootstrap/db-bootstrap';
import { AppModule } from './app.module';
import { HttpExceptionFilter } from './common/filters/http-exception.filter';
import { autoSeed } from './bootstrap/auto-seed';

async function bootstrap() {
  const log = new Logger('Bootstrap');

  // 1) Tự tạo database nếu chưa có
  await ensureDatabase({
    host: process.env.DB_HOST || 'localhost',
    port: +(process.env.DB_PORT || 3306),
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || process.env.DB_PASS || '',
    database: process.env.DB_NAME || 'vton_shop',
  });

  // 2) Khởi động Nest (TypeORM synchronize sẽ tạo bảng tự động)
  const app = await NestFactory.create(AppModule, { bodyParser: true });
  const cs = app.get(ConfigService);

  app.setGlobalPrefix('api', { exclude: ['healthz', 'docs', 'docs-json'] });
  app.useGlobalPipes(
    new ValidationPipe({ whitelist: true, transform: true, forbidNonWhitelisted: false }),
  );
  app.useGlobalFilters(new HttpExceptionFilter());
  app.enableCors({
    origin: cs.get<string>('CORS_ORIGIN', 'http://localhost:5173').split(','),
    credentials: true,
    exposedHeaders: ['X-Pipeline-Info', 'X-Backend', 'X-Warning', 'X-Info'],
  });

  // Uploads dir
  const uploadDir = join(process.cwd(), cs.get<string>('UPLOAD_DIR', 'uploads'));
  mkdirSync(uploadDir, { recursive: true });

  // 3) Swagger
  const swaggerCfg = new DocumentBuilder()
    .setTitle('VTON Shop API')
    .setDescription('Enterprise clothing shop with Virtual Try-On AI')
    .setVersion('0.1.0')
    .addBearerAuth(
      { type: 'http', scheme: 'bearer', bearerFormat: 'JWT', name: 'Authorization', in: 'header' },
      'JWT',
    )
    .addTag('Auth')
    .addTag('Users')
    .addTag('Catalog')
    .addTag('Cart')
    .addTag('Wishlist')
    .addTag('Orders')
    .addTag('Payments')
    .addTag('Reviews')
    .addTag('Sellers')
    .addTag('Admin')
    .addTag('Try-On')
    .build();
  const doc = SwaggerModule.createDocument(app, swaggerCfg);
  SwaggerModule.setup('docs', app, doc, {
    swaggerOptions: { persistAuthorization: true },
    customSiteTitle: 'VTON Shop API Docs',
  });

  // 4) Auto-seed nếu DB rỗng
  if (cs.get<string>('AUTO_SEED', 'true') === 'true') {
    await autoSeed(app).catch((e) => log.warn('auto-seed: ' + e.message));
  }

  const port = +cs.get<number>('PORT', 3000);
  await app.listen(port);
  log.log(`API → http://localhost:${port}/api`);
  log.log(`Swagger → http://localhost:${port}/docs`);
  log.log(`Health → http://localhost:${port}/healthz`);
}
bootstrap();
