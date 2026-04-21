<<<<<<< HEAD
# AI-Image-Enhancer
=======
# Product Image Enhancer

A focused image enhancement component for e-commerce inventory photos. It exposes a FastAPI endpoint and a browser UI where an uploaded image appears immediately, then the enhanced result is displayed beside it for comparison.

## Why This Shape

The component is built around a conservative enhancement pipeline:

1. Validate and normalize uploads.
2. Let the caller choose `RealESRGAN_x4plus` or `pillow_fallback`.
3. Use a deterministic Pillow-based enhancer for local demos and resilience.
4. Return a catalog-ready JPEG as a data URL.

This keeps product details faithful while still improving clarity, contrast, color balance, and perceived sharpness.

## Why Real-ESRGAN Can Look Subtle

Real-ESRGAN is primarily a super-resolution/restoration model. It is most visible on low-resolution, compressed, noisy, or slightly blurry uploads. If the source image is already clean and close to the target output size, the difference can be subtle, especially when both images are displayed in the same browser-sized comparison panel.

This project adds a conservative product-image polish after Real-ESRGAN: color-safe autocontrast, light sharpening, and mild tone adjustment. It intentionally avoids heavy generative edits that could change product details.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Google Colab Notebook

A standalone Colab notebook is available at:

```text
notebooks/image_enhancer_colab.ipynb
```

It installs the inference stack, downloads `RealESRGAN_x4plus.pth`, lets you upload an image, choose `realesrgan` or `pillow_fallback`, compares before/after images, and produces an API-style base64 response.

## API

```http
POST /api/enhance
Content-Type: multipart/form-data

image=<file>
preset=product_standard
engine=pillow_fallback
```

Base64 JSON endpoint:

```http
POST /api/enhance/base64
Content-Type: application/json
```

```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "preset": "product_standard",
  "engine": "pillow_fallback"
}
```

`image_base64` can be a full data URL or a raw base64 string.

Available presets:

- `product_standard`
- `product_detail`
- `product_soft`

Available engines:

- `realesrgan`
- `pillow_fallback`

Check engine readiness:

```http
GET /api/engines
```

## Enabling Real-ESRGAN

The app runs without GPU dependencies by default. For production-quality enhancement, install the optional Real-ESRGAN stack and place the model weight at:

```text
weights/RealESRGAN_x4plus.pth
```

The backend will expose Real-ESRGAN as ready when these imports and weights are available:

- `torch`
- `torchvision`
- `basicsr`
- `realesrgan`
- `numpy`

If they are unavailable, the UI still supports the local fallback enhancer so the frontend and API remain testable.

Example setup:

```bash
pip install -r requirements-realesrgan.txt
pip install --no-deps basicsr==1.4.2 realesrgan==0.3.0
mkdir -p weights
curl -L -o weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

For production, run on a CUDA-capable GPU and keep the fallback path enabled for resilience.
>>>>>>> 530832e (Build image enhancement component)
