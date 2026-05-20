# VTON Shop API (NestJS)

Backend cho hệ thống thương mại thời trang AI Try-On.
**Stack:** NestJS 10 + TypeORM + MySQL 8 + Passport JWT.

## Kiến trúc tổng thể

```
[ React storefront :5173 ]
        │  REST + JWT
        ▼
[ NestJS API :3000 ]  ──► [ MySQL :3306 ]
        │  multipart proxy
        ▼
[ Flask VTON :8000 ]  (giữ nguyên — wrap pipeline AI)
```

API này KHÔNG sửa pipeline AI Python. Nó chỉ proxy request `POST /api/tryon`
sang Flask `app.py` đang chạy ở `:8000`.

## Yêu cầu

- Node.js ≥ 20
- MySQL 8 đang chạy ở localhost:3306
- Flask VTON service đã start ở `:8000` (chạy `python app.py` ở root project)

## Cài đặt

```bash
cd api
cp .env.example .env       # sửa DB_USER, DB_PASS nếu cần
npm install
```

## Tạo database

```sql
CREATE DATABASE vton_shop CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

(`synchronize: true` ở dev → tự tạo bảng khi start.)

## Chạy

```bash
# 1) Start MySQL
# 2) Start Flask VTON
cd ../  &&  python app.py    # cổng 8000

# 3) Start API
cd api  &&  npm run start:dev   # cổng 3000

# 4) Seed dữ liệu mẫu (admin + 3 sản phẩm)
npm run seed

# 5) Start frontend
cd ../web-shop  &&  npm run dev   # cổng 5173 (Vite proxy /api → :3000)
```

Tài khoản admin sau seed: **admin@shop.dev / admin123**

## Endpoint chính (tất cả prefix `/api`)

| Method | Path | Auth | Mô tả |
|---|---|---|---|
| GET  | `/healthz` | — | Healthcheck |
| POST | `/auth/register` | — | Đăng ký khách hàng |
| POST | `/auth/login` | — | Lấy access + refresh token |
| POST | `/auth/refresh` | — | Đổi refresh → access mới |
| GET  | `/auth/me` | JWT | Thông tin user hiện tại |
| GET  | `/users/me` | JWT | Profile + addresses |
| PATCH| `/users/me` | JWT | Cập nhật profile |
| GET  | `/categories` | — | Danh mục |
| GET  | `/products?q=&category=&sort=&page=&size=` | — | Danh sách SP |
| GET  | `/products/:slug` | — | Chi tiết SP |
| POST | `/seller/products` | SELLER/ADMIN | Tạo SP (kèm images, variants, tryonAssets) |
| PUT  | `/seller/products/:id` | SELLER/ADMIN | Sửa SP |
| DELETE | `/seller/products/:id` | SELLER/ADMIN | Xoá |
| POST | `/seller/upload` | SELLER/ADMIN | Upload ảnh → trả `/uploads/xxx.jpg` |
| GET  | `/cart` | JWT | Lấy giỏ |
| POST | `/cart/items` | JWT | Thêm `{variantId, quantity}` |
| PATCH| `/cart/items/:id` | JWT | Sửa số lượng |
| DELETE | `/cart/items/:id` | JWT | Xoá item |
| GET  | `/wishlist` | JWT | DS yêu thích |
| POST | `/wishlist/:productId` | JWT | Thêm yêu thích |
| DELETE | `/wishlist/:productId` | JWT | Bỏ yêu thích |
| POST | `/orders/checkout` | JWT | `{addressId, paymentMethod, note}` |
| GET  | `/orders` | JWT | Đơn của tôi |
| GET  | `/orders/:code` | JWT | Chi tiết đơn |
| GET  | `/payments/vnpay/redirect?code=ORDxxx` | — | Trang VNPay (stub) |
| GET  | `/payments/vnpay/return` | — | Callback VNPay (stub set PAID) |
| GET  | `/products/:productId/reviews` | — | Reviews |
| POST | `/products/:productId/reviews` | JWT | Tạo review |
| POST | `/sellers/apply` | JWT | Đăng ký bán hàng |
| GET  | `/sellers/me` | JWT | Hồ sơ shop |
| GET  | `/admin/stats` | ADMIN | Dashboard |
| GET  | `/admin/users` | ADMIN | DS user |
| PATCH | `/admin/users/:id/status` | ADMIN | Khoá/mở user |
| GET  | `/admin/sellers` | ADMIN | DS seller |
| PATCH | `/admin/sellers/:id/status` | ADMIN | Duyệt seller |
| GET  | `/admin/orders` | ADMIN | DS đơn |
| PATCH | `/admin/orders/:code/status` | ADMIN | Cập nhật trạng thái |
| POST | `/tryon` | optional JWT | **Multipart**: `person`, `cloth` + tham số → proxy Flask `:8000`, stream PNG về |
| GET  | `/tryon/history` | JWT | Lịch sử try-on |

## POST /api/tryon contract

Forward 1:1 sang Flask, giữ nguyên field name. Trả về:
- Body: `image/png` (stream)
- Header: `X-Pipeline-Info`, `X-Backend`, `X-Warning`, `X-Info` (frontend đọc)

Timeout: 20 phút (`VTON_TIMEOUT_MS=1200000`).

## Roles

`ADMIN` > `SELLER` > `CUSTOMER`. Khi `register` mặc định là `CUSTOMER`.
Để có `SELLER`, gọi `POST /sellers/apply` rồi admin duyệt qua `PATCH /admin/sellers/:id/status` body `{status: "APPROVED"}`. (Việc nâng `users.role` lên SELLER có thể làm thủ công hoặc thêm logic sau.)

## Verification

```bash
# 1. Healthcheck
curl http://localhost:3000/healthz

# 2. Đăng ký + login
curl -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"u1@x.com","password":"123456","fullName":"User One"}'

# 3. Test try-on proxy (cần Flask :8000 đang chạy)
curl -X POST http://localhost:3000/api/tryon \
  -F person=@person.jpg -F cloth=@cloth.jpg \
  -F fit_scale=1.12 -F use_catvton_cloud=true \
  -o result.png -D headers.txt
cat headers.txt | grep -i "x-backend\|x-pipeline"
```

## Cấu trúc thư mục

```
api/
  src/
    main.ts, app.module.ts, health.controller.ts, seed.ts
    common/      enums, decorators, guards, filters
    auth/        register, login, JWT, refresh tokens
    users/       profile, addresses
    sellers/     đăng ký bán hàng
    catalog/     products, categories, variants, images, try-on assets
    cart/        cart + items
    wishlist/    
    orders/      checkout, order list, items
    payments/    VNPay stub
    reviews/     
    admin/       dashboard + quản lý user/seller/order
    tryon/       multipart proxy → Flask
  uploads/       ảnh sản phẩm (static, served at /uploads)
```
