<div align="center">

# Xây dựng hệ thống thử đồ ảo dựa trên mô hình thị giác máy tính

### *Virtual Try-On System Powered by Computer Vision Models*

**Hệ thống thử đồ ảo thương mại điện tử end-to-end** — kết hợp diffusion pipeline AI với web shop React và NestJS API.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white">
  <img alt="React" src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white">
  <img alt="NestJS" src="https://img.shields.io/badge/NestJS-10-E0234E?logo=nestjs&logoColor=white">
  <img alt="Tailwind" src="https://img.shields.io/badge/Tailwind-4-38BDF8?logo=tailwindcss&logoColor=white">
  <img alt="Diffusers" src="https://img.shields.io/badge/Diffusers-0.34-FFD21E">
  <img alt="Gemini" src="https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?logo=google&logoColor=white">
</p>

</div>

---

## Tổng quan

Đề tài tốt nghiệp xây dựng một hệ thống Virtual Try-On (VTON) hoàn chỉnh, ứng dụng các mô hình thị giác máy tính (human parsing, pose estimation, garment segmentation, diffusion inpainting) để cho phép người dùng "mặc thử" trang phục trực tuyến trên ảnh của chính mình. Hệ thống được tổ chức thành **ba lớp độc lập** đóng gói trong cùng repository:

| Lớp | Stack | Vai trò |
|---|---|---|
| **AI Pipeline** | Python · PyTorch · Diffusers · OpenCV · MediaPipe | Cloud-primary VTON với fallback local SD-Inpaint |
| **Shop API** | NestJS · TypeORM · MySQL · JWT | E-commerce backend (auth, sản phẩm, giỏ, đơn) |
| **Storefront** | React 19 · Vite · Tailwind 4 · React Router | UI khách hàng + trang thử đồ AI |

Triết lý thiết kế: **TPS warp là output chính, diffusion là refiner** — giữ form và chi tiết áo gốc, chỉ dùng SD để vẽ lại các vùng mép và đổ bóng tự nhiên.

---

## Điểm nổi bật

- **Cloud-primary, local-fallback** — tự động xoay vòng giữa CatVTON · IDM-VTON · Fal.ai FLUX Klein · Replicate, có cooldown và multi-space failover.
- **Pipeline category-aware** — mỗi loại trang phục (`top`/`hoodie`/`jacket`/`pants`/`jeans`/`dress`/`skirt`/`accessory`) có mask, preset và negative-prompt riêng.
- **Dress Pipeline v2** — luồng riêng cho váy với pose-driven silhouette, hair-underlap-aware mask và Telea-inpainted seed (gated bằng `VTON_DRESS_PIPELINE_V2`).
- **Gemini Vision auto-prompt** — phân tích người + áo, sinh positive/negative prompt JSON; có retry + exponential backoff + fallback model + image-hash cache.
- **3 chế độ prompt trên UI**: `Auto Gemini` / `Theo loại` / `Thủ công`.
- **Hoodie subtype** đặc thù — mask siết theo TPS warp, cap `dilate-7px`, prompt chống kink/extra-arm/missing-pocket.
- **Stylist AI — gợi ý phối đồ** — sau khi thử đồ, người dùng chọn dịp/phong cách và gọi Gemini Vision để nhận JSON: màu chính, palette nên phối, màu cần tránh, danh sách item gợi ý (giày · túi · phụ kiện…), style summary và short tip. Endpoint riêng (`POST /api/tryon/recommend`) để không block luồng generate; có cache SHA256 + retry + fallback theo category khi Gemini không khả dụng.
- **Storage tách bạch** — toàn bộ model cache + output đẩy về `VTO_BASE_DIR`, không đụng ổ hệ thống.

---

## Kiến trúc

```
        ┌────────────────────┐         ┌─────────────────────┐
        │  React Storefront  │◄────────│   NestJS Shop API   │
        │  (Vite · TS · TW)  │  REST   │   (TypeORM · MySQL) │
        └─────────┬──────────┘         └─────────────────────┘
                  │ multipart /api/tryon
                  ▼
        ┌────────────────────┐
        │  FastAPI server.py │◄── serves React build + JSON
        └─────────┬──────────┘
                  │ try_on(person, cloth, prompt_mode, …)
                  ▼
        ┌────────────────────────────────────────────────────┐
        │            app.py · pipeline orchestrator           │
        ├────────────────────────────────────────────────────┤
        │  ① Gemini prompt          (auto / fallback / off)  │
        │  ② Garment routing        (category lock + parse)   │
        │  ③ Cloud VTON router      (CatVTON → IDM → FAL)     │
        │         │ on failure                                │
        │         ▼                                           │
        │  ④ Local CPU geometric    (parse · pose · warp)     │
        │  ⑤ Local SD-Inpaint       (dpm++/lcm/hyper-sd)      │
        │  ⑥ Occlusion restore      (hair · hand · shoes)     │
        │  ⑦ Postprocess            (color anchor · ghost rm) │
        └────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
vitrual try on/
├── app.py                       # try_on orchestrator + Gradio UI legacy
├── server.py                    # FastAPI unified server
├── requirements.txt
├── src/
│   ├── pipelines/dress_pipeline.py    # Dress Pipeline v2
│   ├── geometry/                      # Pose-driven silhouettes
│   ├── masks/category_mask_builder.py # Mask theo từng loại trang phục
│   ├── occlusion/                     # Hair / hand / shoes restore
│   ├── postprocess/                   # Color anchor, ghost removal
│   ├── prompts/category_prompts.py    # Positive/negative theo category
│   ├── warps/                         # TPS + affine warp helpers
│   ├── landmarks/                     # Pose smoothing
│   ├── gemini_prompt.py               # Gemini Vision + cache + retry
│   ├── cloud_vton_router.py           # Cloud backend orchestrator
│   ├── catvton_client.py
│   ├── fal_flux_client.py
│   ├── gen_tryon.py                   # SD Inpaint refiner
│   ├── tps_warp.py
│   ├── image_ops.py
│   ├── human_parsing.py               # SegFormer b2_clothes
│   ├── garment_router.py
│   ├── garment_silhouettes.py
│   ├── category_lock.py
│   └── storage.py
├── api/                          # NestJS shop API
│   └── src/                      # auth · products · cart · orders
└── web-shop/                     # React storefront + try-on UI
    ├── src/pages/TryOn.tsx
    ├── src/pages/{Home,ProductDetail,Cart,Checkout,Login}.tsx
    └── src/{components,context,lib}/
```

---

## Bắt đầu nhanh

### Yêu cầu

- Python **3.10+**, Node **18+**, npm
- (Tùy chọn) GPU CUDA cho local diffusion — không bắt buộc, có CPU offload
- (Tùy chọn) MySQL 8 cho NestJS API

### 1 · AI Pipeline + Storefront

```powershell
# Python env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Frontend
cd web-shop && npm install && cd ..

# Chạy unified server (FastAPI + serve React build)
python server.py
```

Mở `http://localhost:8000` — frontend tự build nếu chưa có `web-shop/dist`.

### 2 · Dev mode (hot-reload frontend)

```powershell
# Terminal 1
python server.py

# Terminal 2 — Vite dev proxy /api/* về :8000
cd web-shop
npm run dev      # http://localhost:3000
```

### 3 · NestJS Shop API (tùy chọn)

```powershell
cd api
npm install
npm run start:dev   # http://localhost:3000 (API) — đổi port nếu trùng Vite
```

---

## Cấu hình môi trường

Tạo `.env` ở thư mục gốc:

```env
# Storage — ép tất cả cache về ổ chuyên dụng
VTO_BASE_DIR=E:/virtual_try_on_data
HF_HOME=E:/virtual_try_on_data/huggingface
HUGGINGFACE_HUB_CACHE=E:/virtual_try_on_data/huggingface/hub
TORCH_HOME=E:/virtual_try_on_data/torch

# Cloud VTON
HF_TOKEN=
CATVTON_SPACES=FIT-Check/CatVTON,Nymbo/CatVTON,Shad0ws/CatVTON
IDMVTON_SPACE=yisol/IDM-VTON
FAL_KEY=
FAL_FLUX_LORA_PATH=https://huggingface.co/fal/flux-klein-9b-virtual-tryon-lora/resolve/main/flux-klein-tryon.safetensors
REPLICATE_API_TOKEN=
VTON_CLOUD_TIMEOUT=600
VTON_BACKEND_COOLDOWN_SECONDS=300

# Gemini Vision auto-prompt
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_MODEL_FALLBACK=gemini-2.5-flash-lite
VTON_USE_GEMINI_PROMPT=1
VTON_GEMINI_MAX_RETRIES=3
VTON_GEMINI_BASE_DELAY=1.2
VTON_GEMINI_MAX_DELAY=12
VTON_GEMINI_CACHE=E:/virtual_try_on_data/cache/gemini

# Local SD
HYPERSD_CKPT=Hyper-SD15-8steps-CFG-lora.safetensors
VTON_CPU_OFFLOAD=1
VTON_DRESS_INFER=0

# Dress Pipeline v2 (1 = bật, mặc định)
VTON_DRESS_PIPELINE_V2=1

# Debug
VTON_DEBUG=0
```

---

## API

### `POST /api/tryon`

**Multipart fields**

| Field | Type | Mặc định | Ghi chú |
|---|---|---|---|
| `person` | file | — | Ảnh người mẫu |
| `cloth` | file | — | Ảnh trang phục |
| `cloth_type` | string | `auto` | `auto` · `top` · `hoodie` · `jacket` · `pants` · `jeans` · `dress` · `skirt` · `accessory` |
| `prompt_mode` | string | `auto` | `auto` · `fallback` · `manual` |
| `use_catvton_cloud` | bool | `true` | `false` = bắt buộc local SD |
| `use_gemini_prompt` | bool | `true` | Bật Gemini Vision (chỉ tác dụng khi `prompt_mode=auto`) |
| `style_prompt` | string | `""` | Prompt thủ công, dùng nguyên trong `manual` mode |
| `fit_scale` · `alpha` · `y_offset` | float | preset theo category | Tinh chỉnh fit |
| `gen_steps` · `gen_guidance` · `preserve_strength` | num | preset | Tham số diffusion |
| `quality_preset` | string | `hq` | `fast` · `balanced` · `hq` |
| `refiner_mode` | string | `dpm++` | `lcm` · `hypersd` · `dpm++` · `euler` · `base` |

**Response**: ảnh PNG. Metadata trong headers: `X-Pipeline-Info`, `X-Backend`, `X-Warning`, `X-Info`.

### `POST /api/tryon/describe`

Nhận `person` + `cloth`, trả JSON envelope từ Gemini (hoặc category fallback khi Gemini không khả dụng):

```json
{
  "category": "hoodie",
  "cloth_type": "upper",
  "sleeve_type": "long",
  "neckline": "hood",
  "fit": "regular",
  "silhouette": "fitted",
  "fabric": "cotton fleece",
  "color": "heather grey",
  "positive_prompt": "…",
  "negative_prompt": "…"
}
```

### `GET /api/health`

Health check.

### `POST /api/tryon/recommend`

Stylist AI — nhận ảnh kết quả thử đồ + `category` / `occasion` / `style`, gọi Gemini Vision và trả gợi ý phối đồ. Luôn trả 200 (có fallback theo category khi Gemini lỗi).

**Multipart fields**

| Field | Type | Mặc định | Ghi chú |
|---|---|---|---|
| `result` | file | — | Ảnh PNG/JPG từ `/api/tryon` |
| `category` | string | `garment` | `dress` · `skirt` · `hoodie` · `top` · `jacket` · `pants` · `jeans` · `shorts` … |
| `occasion` | string | `casual` | `đi học` · `đi làm` · `đi chơi` · `hẹn hò` · `dự tiệc` · `du lịch` … |
| `style` | string | `minimal` | `nữ tính` · `streetwear` · `công sở` · `sang trọng` · `Hàn Quốc` · `vintage` … |

**Response**

```json
{
  "success": true,
  "recommendation": {
    "main_color": "be",
    "garment_type": "váy liền thân",
    "suitable_colors": ["trắng", "kem", "nâu", "đen"],
    "avoid_colors": ["xanh neon", "cam chói"],
    "recommended_items": [
      {"type": "shoes", "name": "giày búp bê kem", "reason": "hợp tông, nữ tính"},
      {"type": "bag", "name": "túi nhỏ màu nâu", "reason": "tạo điểm nhấn vừa đủ"}
    ],
    "style_summary": "Thanh lịch, phù hợp đi làm hoặc gặp mặt.",
    "short_tip": "Phối giày kem + túi nâu + phụ kiện vàng nhạt.",
    "source": "gemini"
  }
}
```

`source` có thể là `gemini` · `cache` · `fallback`. Khi Gemini lỗi, response có thêm `warning` mô tả lý do.

---

## Pipeline AI — chi tiết kỹ thuật

### Cloud router (`src/cloud_vton_router.py`)

```
CatVTON multi-space → IDM-VTON → Fal.ai FLUX Klein → Replicate
        │                                              │
        └── cooldown 5 phút khi backend báo lỗi GPU ───┘
```

### Local CPU geometric

`SegFormer b2_clothes` parse 18 nhãn → `MediaPipe` pose → smoothing → `U2Net` (rembg) hoặc fallback threshold + GrabCut → đo vai/hông → prefit scale → affine + TPS warp + sleeve warp theo pose → soft mask blend với hair overlay.

### Local SD-Inpaint (`src/gen_tryon.py`)

- Base: `runwayml/stable-diffusion-inpainting`
- Refiner: `lcm` / `hypersd` / `dpm++` / `euler` / `base`
- LoRA: HyperSD, LCM, optional `CLOTH_LORA_MAP` theo loại trang phục
- Low-VRAM: attention slicing, VAE slicing/tiling, CPU offload, channels-last UNet

### Category-aware masks (`src/masks/category_mask_builder.py`)

Mỗi `category × subtype` có pipeline mask khác nhau. Ví dụ **hoodie**:

- Bỏ `left_arm`/`right_arm` khỏi semantic union (tránh phình cánh tay trần)
- Pose envelope cap về `dilate-5px` của TPS warp
- `MORPH_OPEN 5×5` lên garment_mask để xóa kink tay áo
- Hard cap cuối: `human_prior &= dilate(garment_mask, 7)`

### Gemini prompt với retry (`src/gemini_prompt.py`)

```
analyze_garment_prompt_with_gemini
  ├── SHA256 cache hit? → return ngay
  ├── retry exponential backoff (1.2s → 12s, jitter)
  │      ├── primary  : gemini-2.5-flash
  │      └── fallback : gemini-2.5-flash-lite
  └── on exhaust → caller dùng fallback_describe_garment(category)
```

### Stylist AI — gợi ý phối đồ (`src/gemini_recommend.py`)

Endpoint `POST /api/tryon/recommend` được tách riêng khỏi `/api/tryon` để Gemini latency không block luồng generate (user bấm thêm nút "Gợi ý phối đồ" sau khi đã có ảnh).

```
recommend_outfit_with_gemini(result_rgb, category, occasion, style)
  ├── SHA256 cache key = hash(image) | category | occasion | style | model
  ├── prompt Tiếng Việt yêu cầu JSON schema:
  │     main_color · garment_type · suitable_colors[] · avoid_colors[]
  │     recommended_items[{type, name, reason}] · style_summary · short_tip
  ├── retry + fallback model (giống gemini_prompt)
  └── on failure → fallback_recommendation(category) trả gợi ý mặc định
                   theo dress/skirt · hoodie/top/jacket · pants/jeans/shorts
```

Frontend (`web-shop/src/pages/TryOn.tsx`) render card "Stylist AI" với 2 dropdown (occasion · style), nút "Gợi ý phối đồ", 3 `InfoTile` (màu chính · loại trang phục · nguồn), chip palette nên/không nên phối, grid item gợi ý và tip ngắn.

### Dress Pipeline v2 (`src/pipelines/dress_pipeline.py`)

Luồng độc lập cho váy, gated bằng `VTON_DRESS_PIPELINE_V2=1`:

1. Parse + pose
2. Phân tích silhouette/length/sleeve
3. Pose-driven `target_silhouette` từ width curve
4. Agnostic mask = silhouette ∪ old_clothes ∪ hair_underlap − face − hair_front − shoes
5. Seed bằng Telea inpaint (không flat-color fill)
6. Cloud/local diffusion
7. Restore occluders + ghost removal

---

## Tip kết quả tốt

- Ảnh người: vai/hông rõ, đứng thẳng, ánh sáng đều
- Ảnh trang phục: nền sạch, thấy rõ cổ áo, tay áo, gấu áo
- Hoodie: chọn `cloth_type=hoodie` để kích hoạt mask + prompt riêng
- Váy dài: bật `VTON_DRESS_INFER=768` nếu có GPU ≥ 8GB
- Cloud failing? → đổi `use_catvton_cloud=false` để chạy local

---

## Debug

```powershell
$env:VTON_DEBUG="1"; python server.py
```

Mỗi request tạo `debug_out/<timestamp>_*.png` cho từng giai đoạn pipeline.

---

## Cập nhật gần đây

Tinh chỉnh chất lượng cho **Dress v2** và các subtype top (`jacket`, `tshirt`):

### Dress v2

- **Khử vệt cổ áo cũ** — telea dilation 3 → 9 px để phủ halo viền răng cưa của cổ áo trắng/sơ mi gốc; `_restore_open_neck_skin` nâng alpha 0.58 → 0.92.
- **Vẽ lại được vùng cổ/vai** — composite alpha mở rộng ra `agnostic_mask − target_mask` ở dải ngang hẹp [-8%, +1%] quanh đỉnh váy, chỉ chấp nhận pixel "trông giống da" (r > b+4, lum 80–240, chroma < 90). Tránh dây túi/vạch tối lọt vào.
- **Xác định vai theo váy gốc** — `src/geometry/dress_geometry.py::_shoulder_from_parsing` lấy hàng rộng nhất trong dải 22–42% phía trên `upper_clothes` parsing thay vì MediaPipe (cho kích thước vai thật, hết kéo dài cổ).
- **Đối xứng vai khi bị che** — center theo pose-only (mũi · trung điểm vai · trung điểm hông), `pose_half_lower = |ls.x − rs.x| * 0.5` làm cận dưới cứng. Loại bỏ bias do túi xách che một bên.
- **Hết lộ chân váy cũ** — `remove_old_dress_ghost` thêm forced inpaint pass mọi vùng bên dưới hem mới (bắt được váy sọc/đen mà mean-color filter bỏ sót); kernel close/dilate 5 → 7 px; telea radius 7.
- **Tắt mặc định `_cleanup_offshoulder_top_band`** — gate bằng env `VTON_DRESS_TOPBAND_CLEAN` (mặc định OFF) vì cleanup này hay phủ màu vải lên da gây vệt đỏ ngang vai.

### Jacket subtype

- **`JacketChinCap`** — cắt mask phía trên cằm (chin_y − 1.2% h) để jacket không trườn lên hàm.
- **`JacketHemSeal` v2** — chỉ seal khi PHÁT HIỆN khe hở ≥ 3 cột zeros giữa hai mép mask ở 28% dưới. Kernel 9×41 → 3×11. Giữ lại nét diffusion (răng zip, đường túi).
- **IP-adapter scale 0.58 → 0.48** — nhường chỗ cho SD vẽ chi tiết vải/đường may, đỡ bị paste-y.
- **`JacketInnerShirtRestore`** — sau diffusion, restore pixel áo trong (vd. T-shirt trắng ở cổ V) bằng `person_rgb`: dò vùng `upper_clothes` ở 32% trên có màu cách jacket > 55 (LAB-free RGB distance), loại pixel da, giữ component ≥ 80 px, erode 3×3 + Gaussian σ=1.6 cho mép mềm.

### T-shirt subtype (mới)

- **`TOP_TSHIRT_POSITIVE` / `_CONSTRAINT` / `_NEGATIVE_APPEND`** — khoá họa tiết/chữ in: "preserve front chest graphic exactly: identical wording, font, color, position and scale", "no extra/missing letters". `build_category_negative` xử lý nhánh `tshirt`.
- **`TOP_NEGATIVE_APPEND` mở rộng** — thêm "garbled text, melted letters, smeared print, duplicate logo, recolored graphic, font changed…" áp dụng cho mọi top.
- **IP-adapter scale tshirt = 0.62** — cao hơn shirt (0.52) / hoodie (0.36) để ép diffusion bám reference graphic chặt hơn.

---

## Roadmap

- [ ] Job queue + WebSocket progress thay cho HTTP blocking
- [ ] Upload thật + CMS cho catalog (thay `products.ts` mock)
- [ ] A/B telemetry giữa cloud backends
- [ ] Test suite cho mask builder + warp helpers
- [ ] Trang admin/seller hoàn thiện trong NestJS
- [ ] Đóng gói Docker compose 3 service (Python · Nest · MySQL)

---

## Mô hình tham khảo (Reference Models)

Hệ thống tổ hợp nhiều mô hình thị giác máy tính — mỗi mô hình đóng vai trò khác nhau trong pipeline:

### Cloud VTON backends

| Mô hình | Nhà phát hành | Vai trò trong project |
|---|---|---|
| **CatVTON** | Zheng-Chong / FIT-Check (HF Spaces) | Backend mặc định — diffusion-based virtual try-on, in-context attention, multi-space failover |
| **IDM-VTON** | yisol (HF Spaces) | Backend phụ — improved diffusion, mạnh ở váy dài + áo phức tạp |
| **FLUX Klein Try-On LoRA** | fal-ai | Cloud refiner FLUX + LoRA chuyên cho try-on |
| **Replicate VTON** | Replicate registry | Fallback cuối cùng khi 3 backend trên busy/quota |

### Local diffusion stack

| Mô hình | Nhà phát hành | Vai trò |
|---|---|---|
| **Stable Diffusion v1.5 Inpainting** (`runwayml/stable-diffusion-inpainting`) | Runway / Stability AI | Base inpainting model cho refiner local |
| **Hyper-SD LoRA** (`ByteDance/Hyper-SD`, 8-step CFG) | ByteDance | Tăng tốc inference còn 8 step, giữ chất lượng |
| **LCM-LoRA** (`latent-consistency/lcm-lora-sdv1-5`) | Tsinghua / Hugging Face | Latent Consistency, preset `lcm` |
| **IP-Adapter** (`h94/IP-Adapter`, `ip-adapter-plus_sd15`) | Tencent AI Lab | Image-prompt conditioning từ ảnh trang phục |
| **DPM++ / Euler schedulers** | Diffusers | Refiner mode cao cấp cho preset `hq` |

### Human parsing · pose · segmentation

| Mô hình | Nhà phát hành | Vai trò |
|---|---|---|
| **SegFormer B2 Clothes** (`mattmdjaga/segformer_b2_clothes`) | Matt Mdjaga / NVIDIA SegFormer | Human parsing 18 nhãn (hair, face, upper, pants, dress…) |
| **MediaPipe Pose** | Google MediaPipe | Pose estimation 33 keypoints, vai/hông/khớp tay |
| **U²-Net (rembg)** (`u2net.onnx`) | Qin et al. | Garment background removal |
| **OpenCV GrabCut** | OpenCV | Fallback segmentation khi U²-Net fail |

### Vision-Language

| Mô hình | Nhà phát hành | Vai trò |
|---|---|---|
| **Gemini 2.5 Flash / Flash-Lite** | Google DeepMind | Auto-prompt generator + Stylist AI recommend, fallback model chain |

### Thuật toán cổ điển

- **Thin Plate Spline (TPS) Warp** — Bookstein 1989 — warp 20 control point + symmetry, là output chính của pipeline.
- **Telea Inpainting** (`cv2.inpaint INPAINT_TELEA`) — Telea 2004 — seed fill cho Dress Pipeline v2, ghost removal.
- **GrabCut** — Rother et al. 2004 — fallback foreground/background extraction.

---

## Tài liệu tham khảo (References)

### Papers

1. Bookstein, F. L. (1989). *Principal Warps: Thin-Plate Splines and the Decomposition of Deformations*. IEEE PAMI 11(6).
2. Chong, Z. et al. (2024). *CatVTON: Concatenation Is All You Need for Virtual Try-On*. arXiv:2407.15886.
3. Choi, Y. et al. (2024). *IDM-VTON: Improving Diffusion Models for Authentic Virtual Try-on in the Wild*. ECCV 2024 / arXiv:2403.05139.
4. Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models* (Stable Diffusion). CVPR.
5. Xie, E. et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS.
6. Ye, H. et al. (2023). *IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models*. arXiv:2308.06721.
7. Luo, S. et al. (2023). *LCM-LoRA: A Universal Stable-Diffusion Acceleration Module*. arXiv:2311.05556.
8. Ren, Y. et al. (2024). *Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis*. arXiv:2404.13686.
9. Qin, X. et al. (2020). *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection*. Pattern Recognition.
10. Rother, C., Kolmogorov, V., Blake, A. (2004). *GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts*. ACM TOG.
11. Telea, A. (2004). *An Image Inpainting Technique Based on the Fast Marching Method*. Journal of Graphics Tools.

### Repositories & dự án nguồn

- [Zheng-Chong/CatVTON](https://github.com/Zheng-Chong/CatVTON) — base implementation + HF Space.
- [yisol/IDM-VTON](https://github.com/yisol/IDM-VTON) — improved diffusion VTON.
- [fal-ai/flux-klein-tryon-lora](https://huggingface.co/fal/flux-klein-9b-virtual-tryon-lora) — FLUX LoRA cho try-on.
- [huggingface/diffusers](https://github.com/huggingface/diffusers) — Stable Diffusion + Inpaint + LoRA loader.
- [tencent-ailab/IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) — image prompt adapter.
- [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) — 8-step LoRA.
- [latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) — LCM LoRA.
- [mattmdjaga/segformer_b2_clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) — human parsing.
- [google/mediapipe](https://github.com/google/mediapipe) — Pose Landmarker.
- [danielgatis/rembg](https://github.com/danielgatis/rembg) — U²-Net inference wrapper.
- [google-gemini/cookbook](https://github.com/google-gemini/cookbook) — Gemini Vision usage patterns.

### Documentation chính thức

- [PyTorch](https://pytorch.org/docs) · [Diffusers](https://huggingface.co/docs/diffusers) · [Transformers](https://huggingface.co/docs/transformers)
- [FastAPI](https://fastapi.tiangolo.com) · [NestJS](https://docs.nestjs.com) · [TypeORM](https://typeorm.io)
- [React](https://react.dev) · [Vite](https://vitejs.dev) · [Tailwind CSS](https://tailwindcss.com)
- [OpenCV](https://docs.opencv.org) · [MediaPipe Solutions](https://developers.google.com/mediapipe)
- [Google Gen AI SDK](https://ai.google.dev/gemini-api/docs)

---

## License

Đồ án tốt nghiệp — sử dụng cho mục đích học thuật.
