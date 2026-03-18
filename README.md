# Đồ án: Hệ thống Virtual Try-On (tối ưu cho máy VRAM thấp)

README này mô tả phiên bản pipeline hiện tại đã được tinh chỉnh nhiều vòng để sửa các lỗi thực tế trong thử đồ ảo:

- cổ áo cũ còn sót
- áo bị bó eo/ngực
- tay áo và nách méo
- áo mới bám theo shape áo cũ của người mẫu (sports bra/crop top)
- texture áo bị bẩn khi refine

Mục tiêu của hệ thống là chạy ổn trên máy yếu (CPU hoặc GPU VRAM thấp), nhưng vẫn giữ chất lượng try-on tốt cho đồ án.

---

## 1) Tổng quan kiến trúc

Pipeline hiện tại trong UI:

`Parsing -> Pose Smoothing -> U2Net Cloth Seg -> Body Measurement -> Shape-Constrained PreFit -> Affine Align + TPS Warp -> Skeleton Erase -> Skin Restore -> Edge Feather -> Poisson Blend -> Layer Compositing -> Edge-Only Masked Diffusion (optional)`

Ý tưởng chính:

- Dùng **pose/skeleton** làm trục hình học chính (không warp theo shape áo cũ trên người mẫu).
- Dùng **hybrid cloth segmentation** để tránh thiếu mask vùng cổ/bụng.
- Dùng **shape constraint** để giữ form áo (không bị bó vào torso).
- Dùng **edge-only diffusion** để sửa mép/cổ/tay tự nhiên mà không phá texture lõi.

---

## 2) Tính năng chính

- Web app bằng **Gradio**.
- Input: ảnh người mẫu + ảnh áo/quần.
- Output: ảnh kết quả và lưu tự động vào ổ E.
- Pose detection: **MediaPipe**.
- Human parsing: **SegFormer** (`mattmdjaga/segformer_b2_clothes`).
- Cloth segmentation: **U2Net (rembg)** + fallback + merge với parsing.
- Warp hình học: **Affine pre-align + TPS refine** (9 landmarks).
- Blending: edge feather + Poisson seamless clone.
- Optional refinement: **Stable Diffusion Inpainting** (`lcm`, `hypersd`, `dpm++`, `euler`, `base`).
- Optional cloud route: `CatVTON -> IDM-VTON` với fallback tự động.

---

## 3) Các chỉnh sửa đã áp dụng (changelog kỹ thuật)

### 3.1 Geometry / Warp

- Chuyển từ warp đơn giản sang **Affine + TPS** để ổn định vai/cổ trước khi biến dạng cục bộ.
- TPS dùng 9 điểm landmark: `collar`, `shoulder_l/r`, `armpit_l/r`, `mid_l/r`, `hem_l/r`.
- Tăng regularization TPS để giảm kéo méo quá mức.
- Bổ sung **Garment Shape Constraint** trong prefit:
  - scale theo `max(shoulder_width, hip_width)` thay vì chỉ shoulder
  - giữ tối thiểu form gốc bằng `preserve_ratio`
- Khóa chiều dài áo theo torso (`target_h` dựa trên `torso_height`) để tránh thành crop-top.

### 3.2 Remove old cloth / body cleaning

- Thay cách erase theo parsing áo cũ bằng **Skeleton Erase**:
  - tạo mask xóa full torso từ vai tới hông
  - mở rộng qua vùng nách/tay
  - union với vùng `warped_mask` để xóa đủ nơi áo mới sẽ phủ
- Neck region được cưỡng bức erase để tránh sót cổ áo cũ.

### 3.3 Segmentation

- Thêm **hybrid segmentation cho ảnh áo**:
  - `mask_u2net = segment_cloth_u2net(...)`
  - `mask_segformer = parse_human(cloth_rgb) -> get_clothing_mask(...)`
  - merge: `OR(mask_u2net, mask_segformer)` nếu segformer hợp lệ
- Morph close + dilate để giảm thiếu mask vùng bụng/cổ/tay.

### 3.4 Diffusion refine

- Dùng **edge-only diffusion** thay vì full-garment diffusion.
- Edge mask được mở rộng theo kinh nghiệm VTON:
  - `mask_dilate ~= 12`
  - `mask_blur ~= 8` (đang dùng kernel blur gần tương đương)
- Mục tiêu: diffusion tập trung sửa viền/cổ/tay, không phá texture lõi.

### 3.5 Compositing safeguards

- Ngăn `SkinRestore` đè ngược lên vùng áo warp (tránh lộ bụng giả khi blend).
- Bảo vệ tay áo khi layer foreground arms/face/hair.

---

## 4) Thông số mặc định hiện tại (UI)

- `Độ rộng trang phục`: `1.12`
- `Độ hòa trộn`: `0.65`
- `Dịch dọc`: `-0.01`
- `Refine steps`: `18`
- `Refine guidance`: `1.3`
- `Giữ texture áo gốc`: `0.80`
- `Preset`: `hq`
- `Refiner mode`: `dpm++`

Gợi ý cho máy yếu hơn:

- `refiner_mode = lcm`
- `steps = 6-12`
- `guidance = 1.0-2.0`

---

## 5) Cài đặt và chạy

### 5.1 Tạo môi trường

```powershell
cd "e:\vitrual try on"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 5.2 Chạy ứng dụng

```powershell
python app.py
```

Mở: `http://127.0.0.1:7860`

---

## 6) Cấu hình lưu trữ và cache

Mặc định lưu tại:

- `E:/virtual_try_on_data/inputs`
- `E:/virtual_try_on_data/outputs`
- `E:/virtual_try_on_data/cache`

Biến môi trường liên quan:

- `VTO_BASE_DIR`
- `HF_HOME`
- `TRANSFORMERS_CACHE` (legacy, có thể chuyển dần sang `HF_HOME`)
- `HUGGINGFACE_HUB_CACHE`
- `TORCH_HOME`

---

## 7) Cloud backend và biến môi trường

Cloud VTON Router (tùy chọn) sẽ thử theo thứ tự:

- `CatVTON`
- `IDM-VTON`

Biến môi trường hữu ích:

- `HF_TOKEN`
- `CATVTON_SPACE`
- `IDMVTON_SPACE`
- `VTON_BACKEND_COOLDOWN_SECONDS`
- `HYPERSD_CKPT`
- `CLOTH_LORA_MAP`
- `CLOTH_LORA_SCALE`

---

## 8) Các mô hình thử đồ đã tham khảo và nghiên cứu

Các mô hình/hướng đã được tham khảo trong quá trình làm đồ án:

- `CP-VTON / CP-VTON+`
  - Vai trò: baseline kinh điển cho hướng geometric matching + warping.
  - Điểm học được: tách bước matching/warp trước khi synthesis giúp kiểm soát form áo tốt.
- `VITON-HD`
  - Vai trò: tham chiếu cho hướng high-resolution virtual try-on.
  - Điểm học được: cần parsing tốt và compositing/layering cẩn thận để giảm artefact biên.
- `HR-VITON`
  - Vai trò: tham chiếu cho pipeline warping + refinement chất lượng cao.
  - Điểm học được: mask refinement và bảo toàn biên áo là yếu tố quyết định realism.
- `IDM-VTON`
  - Vai trò: tham chiếu cho hướng diffusion-based try-on hiện đại.
  - Điểm học được: edge-aware refine và text/condition control giúp kết quả tự nhiên hơn.
- `CatVTON`
  - Vai trò: backend cloud VTON chuyên dụng để nâng chất lượng khi local chưa đủ mạnh.
  - Trạng thái trong đồ án: đã tích hợp router cloud theo thứ tự `CatVTON -> IDM-VTON`.
- `Stable Diffusion Inpainting + LCM/HyperSD/DPM++/Euler`
  - Vai trò: local masked refiner chạy được trên máy VRAM thấp.
  - Trạng thái trong đồ án: đang dùng để refine mép/cổ/tay với edge-only mask.
- `Flux Klein Try-On LoRA (fal.ai)`
  - Vai trò: backend API nâng chất lượng khi cần.
  - Trạng thái trong đồ án: hỗ trợ qua biến môi trường và route cloud.

Các mô hình parsing/segmentation liên quan đã tham khảo:

- `SegFormer (mattmdjaga/segformer_b2_clothes)`
  - Trạng thái: đang dùng cho human parsing trong pipeline.
- `U2Net (rembg)`
  - Trạng thái: đang dùng cho cloth segmentation, có fallback và merge mask.
- `SCHP`
  - Trạng thái: đã nghiên cứu, chưa tích hợp chính thức (đặt trong hướng nâng cấp tiếp theo).

---

## 9) Cấu trúc mã nguồn

- `app.py`: orchestrator pipeline + UI Gradio.
- `src/image_ops.py`: pose, mask, prefit, erase, blend utilities.
- `src/tps_warp.py`: landmark detection + Affine/TPS warp.
- `src/human_parsing.py`: SegFormer parsing + helper masks.
- `src/gen_tryon.py`: masked diffusion refiner và scheduler.
- `src/cloud_vton_router.py`: cloud backend routing/fallback.
- `src/storage.py`: đường dẫn output/cache.

---

## 10) Hướng nâng cấp tiếp theo

- Tích hợp **SCHP parsing** như một nhánh segmentation thay thế/bổ sung SegFormer.
- Garment landmark detection chuyên biệt (collar/hem/sleeve endpoints).
- Adaptive warp strength theo vùng (shoulder/chest/waist) thay vì hằng số toàn cục.
- Edge diffusion mask học theo confidence map thay vì kernel cố định.

---

## 11) Lưu ý chất lượng input

Để cho kết quả tốt nhất:

- Ảnh áo nền trắng/sạch, ít nếp gấp mạnh.
- Ảnh người rõ vai-hông-tay, không che torso.
- Tránh ảnh người có pose quá nghiêng hoặc crop mất phần hông.

Điều này giúp các bước parsing + warp + blend hoạt động ổn định hơn.
