# VTON Shop - Hệ thống Virtual Try-On

Project này là đồ án thử đồ ảo cho thời trang, gồm một web shop React và một backend AI xử lý ảnh người mẫu + ảnh trang phục để tạo ảnh try-on. Trọng tâm hiện tại là kiến trúc cloud-primary: ưu tiên dùng các backend VTON trên cloud để có chất lượng tốt hơn, sau đó tự động fallback về pipeline local khi cloud không khả dụng.

## Những gì đã đạt được

### 1. Web shop hoàn chỉnh ở mức prototype

- Đã xây dựng frontend bằng React, TypeScript, Vite và Tailwind CSS.
- Có trang danh sách sản phẩm với tìm kiếm, lọc theo danh mục và sắp xếp theo giá hoặc đánh giá.
- Có trang chi tiết sản phẩm với ảnh, giá, mô tả, size, màu sắc, rating và nút thử đồ AI.
- Có giỏ hàng bằng React Context: thêm sản phẩm, tăng/giảm số lượng, xóa từng dòng, xóa toàn bộ và tính tổng tiền.
- Có routing đầy đủ:
  - `/` - danh sách sản phẩm
  - `/product/:id` - chi tiết sản phẩm
  - `/try-on` - thử đồ bằng ảnh tự upload
  - `/try-on/:id` - thử đồ từ sản phẩm đã chọn
  - `/cart` - giỏ hàng
- Có giao diện responsive cho desktop và mobile.

### 2. Trang thử đồ AI trên frontend

- Người dùng có thể upload ảnh người mẫu và ảnh trang phục.
- Nếu đi từ trang sản phẩm, ảnh trang phục được tự động lấy từ sản phẩm đó.
- Có hai chế độ xử lý:
  - `Cloud AI`: ưu tiên CatVTON / IDM-VTON / Fal.ai.
  - `Local SD`: dùng pipeline local với Stable Diffusion Inpainting.
- Có nhóm tùy chỉnh pipeline:
  - độ rộng trang phục
  - độ hòa trộn
  - dịch dọc
  - prompt mô tả trang phục
  - số bước refine
  - guidance
  - mức giữ texture gốc
  - preset chất lượng
  - refiner mode
  - loại trang phục
- Frontend gọi API `/api/tryon`, nhận kết quả PNG, hiển thị trạng thái xử lý, backend đã dùng, warning và pipeline info.
- Có thao tác reset và tải ảnh kết quả.

### 3. Backend FastAPI thống nhất

- Đã có `server.py` để chạy một process duy nhất cho cả frontend và API.
- Backend cung cấp:
  - `POST /api/tryon`: nhận multipart upload `person` và `cloth`, chạy pipeline, trả ảnh PNG.
  - `GET /api/health`: health check.
- Khi có build frontend ở `web-shop/dist`, server tự serve SPA React.
- Nếu chưa có `dist`, `server.py` có logic build frontend bằng `npm run build` trước khi khởi động.
- API trả metadata qua header:
  - `X-Info`
  - `X-Pipeline-Info`
  - `X-Backend`
  - `X-Warning`
- Có timeout xử lý qua `VTON_CLOUD_TIMEOUT`, mặc định 600 giây.
- Có CORS để frontend dev server có thể gọi API.

### 4. Gradio UI vẫn được giữ lại

- `app.py` vẫn có giao diện Gradio standalone để test pipeline trực tiếp.
- Chạy bằng `python app.py`, mặc định mở ở `http://127.0.0.1:7860`.
- Gradio dùng cùng hàm `try_on()` với FastAPI, nên thuận tiện để debug pipeline mà không cần frontend React.

### 5. Pipeline AI cloud-primary

Pipeline chính trong `try_on()` hiện đi theo hướng:

```text
Input person + cloth
-> lưu ảnh tạm
-> phân loại trang phục
-> Cloud VTON nếu bật
-> hậu xử lý giữ identity
-> nếu cloud lỗi: fallback local CPU geometric pipeline
-> nếu bật AI refine: local Stable Diffusion Inpainting
-> overlay lại tóc / foreground khi có parsing
-> lưu output
```

Cloud router đã đạt được:

- Tự động thử nhiều backend theo thứ tự:
  1. CatVTON Cloud
  2. IDM-VTON Cloud
  3. Fal.ai FLUX Klein Try-On LoRA
  4. Replicate, nếu cấu hình token
- Có cooldown cho backend cloud bị lỗi GPU, paused hoặc runtime error.
- Có CatVTON multi-space failover qua `CATVTON_SPACES`.
- Có hỗ trợ `HF_TOKEN` cho Hugging Face Space private hoặc cần token.
- Có fallback local nếu toàn bộ cloud route không khả dụng.

### 6. Pipeline local CPU geometric fallback

Pipeline local đã được nâng cấp nhiều phần để tránh phụ thuộc hoàn toàn vào cloud:

- Human parsing bằng SegFormer `mattmdjaga/segformer_b2_clothes`.
- Pose detection bằng MediaPipe.
- Làm mượt landmark pose để giảm méo khi warp.
- Cloth segmentation bằng U2Net qua `rembg` nếu có, fallback về threshold + GrabCut nếu thiếu thư viện.
- Merge thêm mask từ SegFormer khi parse được ảnh trang phục.
- Body measurement để đo vai, hông, torso và chân.
- Pre-fit scale trang phục trước khi warp để giữ form.
- Phân loại trang phục theo silhouette:
  - top
  - pants
  - dress
- Phân loại sleeve:
  - sleeveless
  - short sleeve
  - long sleeve
- Có xử lý riêng cho pants bằng hip alignment và landmark phần chân.
- Có xử lý riêng cho dress bằng full-body coverage, dress mask và các bước phục hồi texture.
- Warp bằng affine + TPS, có thêm sleeve warp theo pose tay.
- Erase áo cũ bằng skeleton mask thay vì chỉ dựa vào parsing cũ.
- Blend bằng soft mask và bảo vệ foreground như tay, mặt, tóc.
- Overlay lại tóc gốc sau khi tạo ảnh để giữ identity tốt hơn.

### 7. Local diffusion refinement

Khi bật AI refinement mà cloud không dùng được hoặc người dùng chọn Local SD, project dùng Stable Diffusion Inpainting:

- Model chính: `runwayml/stable-diffusion-inpainting`.
- Có các refiner mode:
  - `lcm`
  - `hypersd`
  - `dpm++`
  - `euler`
  - `base`
- Có hỗ trợ LCM LoRA và Hyper-SD LoRA.
- Có optional cloth-type LoRA qua `CLOTH_LORA_MAP`.
- Có tối ưu cho máy VRAM thấp:
  - attention slicing
  - VAE slicing
  - VAE tiling
  - model CPU offload
  - channels-last cho UNet khi dùng CUDA
- Mặc định inference size 512 để phù hợp máy yếu hơn; dress có thể tăng bằng `VTON_DRESS_INFER`.
- Có cơ chế clamp/sanitize NaN, Inf và ảnh đầu vào trước khi đưa vào diffusion.

### 8. Lưu trữ, cache và debug

- Đã tách cấu hình lưu trữ vào `src/storage.py`.
- Mặc định lưu toàn bộ dữ liệu tại:

```text
E:/virtual_try_on_data
```

- Các thư mục tự tạo:
  - `inputs`
  - `outputs`
  - `cache`
  - `huggingface`
  - `huggingface/hub`
  - `torch`
- Output được lưu dạng:

```text
tryon_YYYYMMDD_HHMMSS.png
```

- Có `VTON_DEBUG=1` để lưu ảnh trung gian vào thư mục debug.
- Cache Hugging Face và Torch được ép về ổ cấu hình để tránh đầy ổ hệ thống.

## Kiến trúc tổng quan

```text
web-shop/ React frontend
        |
        | POST /api/tryon
        v
server.py FastAPI unified server
        |
        v
app.py try_on orchestrator
        |
        +--> Cloud router
        |       +--> CatVTON
        |       +--> IDM-VTON
        |       +--> Fal.ai FLUX
        |       +--> Replicate
        |
        +--> Local fallback
                +--> MediaPipe pose
                +--> SegFormer human parsing
                +--> U2Net / fallback cloth mask
                +--> affine + TPS warp
                +--> local SD inpainting refine
                +--> compositing + hair overlay
```

## Công nghệ sử dụng

### Backend / AI

- Python
- FastAPI
- Gradio
- OpenCV
- NumPy
- Pillow
- MediaPipe
- PyTorch
- Transformers
- Diffusers
- Accelerate
- Safetensors
- PEFT
- Fal client
- Gradio client

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- React Router
- Lucide React

## Cài đặt

### 1. Tạo môi trường Python

```powershell
cd "e:\vitrual try on"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nếu muốn dùng U2Net qua `rembg`, có thể cài thêm:

```powershell
pip install rembg onnxruntime
```

Nếu không cài `rembg`, project vẫn có fallback segmentation.

### 2. Cài frontend

```powershell
cd web-shop
npm install
cd ..
```

## Cách chạy

### Chạy full-stack bằng một server

```powershell
python server.py
```

Server sẽ chạy tại:

```text
http://localhost:8000
```

Các đường dẫn chính:

- Frontend: `http://localhost:8000`
- Try-on page: `http://localhost:8000/try-on`
- API: `http://localhost:8000/api/tryon`
- Health check: `http://localhost:8000/api/health`

### Chạy frontend dev server

Trong terminal 1:

```powershell
python server.py
```

Trong terminal 2:

```powershell
cd web-shop
npm run dev
```

Vite chạy ở:

```text
http://localhost:3000
```

`vite.config.ts` đã proxy `/api/tryon` về `http://127.0.0.1:8000`.

### Chạy Gradio standalone

```powershell
python app.py
```

Mở:

```text
http://127.0.0.1:7860
```

## Biến môi trường quan trọng

Có thể tạo file `.env` dựa trên `.env.example`.

### Storage và cache

```env
VTO_BASE_DIR=E:/virtual_try_on_data
HF_HOME=E:/virtual_try_on_data/huggingface
HUGGINGFACE_HUB_CACHE=E:/virtual_try_on_data/huggingface/hub
TORCH_HOME=E:/virtual_try_on_data/torch
```

### Cloud backend

```env
HF_TOKEN=
HUGGINGFACEHUB_API_TOKEN=
CATVTON_SPACES=FIT-Check/CatVTON,Nymbo/CatVTON,Shad0ws/CatVTON,hungdang1610/CatVTON
IDMVTON_SPACE=yisol/IDM-VTON
FAL_KEY=
FAL_API_KEY=
FAL_FLUX_LORA_PATH=https://huggingface.co/fal/flux-klein-9b-virtual-tryon-lora/resolve/main/flux-klein-tryon.safetensors
REPLICATE_API_TOKEN=
REPLICATE_VTON_MODEL=cuuupid/idm-vton:latest
VTON_BACKEND_COOLDOWN_SECONDS=300
VTON_CLOUD_TIMEOUT=600
```

### Local diffusion

```env
HYPERSD_CKPT=Hyper-SD15-8steps-CFG-lora.safetensors
CLOTH_LORA_MAP=
CLOTH_LORA_SCALE=0.65
VTON_FORCE_FP32=0
VTON_CPU_OFFLOAD=1
VTON_DRESS_INFER=0
VTON_DRESS_FAST_REFINER=0
VTON_DRESS_DIFFUSION_PRIMARY=0
VTON_POST_HAIR_RED_CLEAN=0
```

### Debug

```env
VTON_DEBUG=1
```

## Cấu trúc mã nguồn

```text
.
├── app.py                       # Orchestrator pipeline + Gradio UI
├── server.py                    # FastAPI server + serve React build
├── requirements.txt             # Python dependencies
├── .env.example                 # Mẫu cấu hình môi trường
├── src/
│   ├── image_ops.py             # Pose, mask, erase, blend, prefit utilities
│   ├── human_parsing.py         # SegFormer human parsing
│   ├── tps_warp.py              # Affine/TPS warp, garment/pants/dress logic
│   ├── gen_tryon.py             # Stable Diffusion inpainting refiner
│   ├── cloud_vton_router.py     # Cloud backend router/fallback
│   ├── catvton_client.py        # CatVTON client
│   ├── fal_flux_client.py       # Fal.ai FLUX client
│   └── storage.py               # Output/cache path management
└── web-shop/
    ├── package.json
    ├── vite.config.ts
    └── src/
        ├── App.tsx
        ├── pages/
        │   ├── Home.tsx
        │   ├── ProductDetail.tsx
        │   ├── TryOn.tsx
        │   └── Cart.tsx
        ├── components/
        │   ├── Navbar.tsx
        │   ├── Footer.tsx
        │   └── ProductCard.tsx
        ├── context/
        │   └── CartContext.tsx
        └── data/
            └── products.ts
```

## API `/api/tryon`

Endpoint nhận `multipart/form-data`.

### File inputs

- `person`: ảnh người mẫu.
- `cloth`: ảnh trang phục.

### Form fields

- `fit_scale`
- `alpha`
- `y_offset`
- `use_gen`
- `style_prompt`
- `gen_steps`
- `gen_guidance`
- `preserve_strength`
- `quality_preset`
- `refiner_mode`
- `cloth_type`
- `use_catvton_cloud`

Response thành công là ảnh PNG. Metadata nằm trong response headers.

## Lưu ý chất lượng input

Để kết quả ổn định hơn:

- Ảnh người nên rõ vai, hông và tay.
- Tránh pose quá nghiêng hoặc crop mất phần thân.
- Ảnh trang phục nên nền sạch, ít nhăn mạnh, thấy rõ cổ áo, tay áo và gấu áo.
- Với quần hoặc váy/dress, ảnh người nên có đủ phần hông và chân.
- Cloud backend cần internet và có thể phụ thuộc trạng thái GPU của Hugging Face Spaces.
- Local diffusion lần đầu sẽ tải model, thời gian chạy phụ thuộc GPU/CPU.

## Hướng phát triển tiếp theo

- Sửa encoding tiếng Việt trong một số file frontend đang bị hiển thị mojibake khi đọc bằng PowerShell mặc định.
- Bổ sung test tự động cho API và các helper xử lý mask/warp.
- Chuẩn hóa `.env.example` để bao phủ toàn bộ biến cloud hiện đang dùng.
- Thêm queue/job status cho request try-on dài thay vì chờ một HTTP request đơn.
- Thêm upload thật cho catalog sản phẩm thay vì dữ liệu mock trong `products.ts`.
- Tối ưu UX trạng thái cloud fallback để người dùng biết backend nào đang chạy.
- Tinh chỉnh riêng cho từng loại trang phục: áo khoác, hoodie, váy dài, quần short và jeans.
