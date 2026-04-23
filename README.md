# Product Image Enhancer

A focused image enhancement component for e-commerce inventory photos. It exposes a FastAPI endpoint and a browser UI where an uploaded image appears immediately, then the enhanced result is displayed beside it for comparison.

## Why This Shape

The component is built around a conservative enhancement pipeline:

1. Validate and normalize uploads.
2. Let the caller choose `studio_product_generative`, `studio_product_flux_4bit`, `studio_product_flux`, `studio_product_focus`, `studio_product`, `RealESRGAN_x4plus`, or `pillow_fallback`.
3. Use a deterministic Pillow-based enhancer for local demos and resilience.
4. Return a catalog-ready JPEG as a data URL.

This keeps product details faithful while still improving clarity, contrast, color balance, and perceived sharpness.

## Why Real-ESRGAN Can Look Subtle

Real-ESRGAN is primarily a super-resolution/restoration model. It is most visible on low-resolution, compressed, noisy, or slightly blurry uploads. If the source image is already clean and close to the target output size, the difference can be subtle, especially when both images are displayed in the same browser-sized comparison panel.

This project adds a conservative product-image polish after Real-ESRGAN: color-safe autocontrast, light sharpening, and mild tone adjustment. It intentionally avoids heavy generative edits that could change product details.

## Studio Product Mode

`studio_product` is the non-generative catalog-photo mode. It uses rembg segmentation when available, places the product on a square studio canvas, adds a soft contact shadow, and applies safe product polish. The first request may download the rembg `isnet-general-use` segmentation model into `weights/rembg`.

If rembg is unavailable, the app falls back to stricter OpenCV GrabCut masking. If masking is uncertain, the engine keeps a polished full image rather than producing a bad cutout. For tall apparel images where segmentation is likely cutting away part of the garment, it switches to a safer framed studio crop so the product is preserved.

Output sizing preserves the uploaded image's original long edge when it is larger than the configured target. `target_long_edge` is now a minimum quality target, while `MAX_OUTPUT_LONG_EDGE` defaults to `4096` as a safety cap for very large uploads.

`studio_product_focus` uses the same foreground-detection stack, but keeps the original scene instead of replacing the background. It enhances the detected product area with a soft mask while leaving the background present, so busy store photos can look cleaner without introducing cutout artifacts.

`studio_product` is not a trainable model yet; it is a deterministic pipeline. Real store images are still useful because they reveal failure cases and let us tune the masking and fallback rules.

Batch-check local store images with:

```bash
python scripts/evaluate_studio_product.py "ALATPay Store Images" --output-dir outputs/alatpay_studio_eval
```

The script writes enhanced outputs and `comparison_sheet.jpg` for quick visual review. Local store-image folders and generated outputs are ignored by git.

## Studio Product Flux Mode

`studio_product_flux_4bit` is the Colab-friendly open-weight generative test engine. It first builds the safer `studio_product` reference image, then runs Flux image-to-image with the compact pre-quantized `magespace/FLUX.1-schnell-bnb-nf4` checkpoint and the same strict product-preservation prompt used by the controlled generative mode.

`studio_product_flux` uses the full `black-forest-labs/FLUX.1-schnell` Diffusers checkpoint. It has the most direct upstream compatibility, but the download is much larger.

Install the optional local dependencies with:

```bash
pip install -r requirements-flux.txt
```

Recommended Colab settings:

```bash
export FLUX_MODEL_ID=black-forest-labs/FLUX.1-schnell
export FLUX_4BIT_MODEL_ID=magespace/FLUX.1-schnell-bnb-nf4
export FLUX_IMAGE_SIZE=768
export FLUX_4BIT_IMAGE_SIZE=768
export FLUX_NUM_INFERENCE_STEPS=4
export FLUX_STRENGTH=0.55
export FLUX_GUIDANCE_SCALE=0.0
```

Flux is much heavier than the deterministic engines. Use a CUDA GPU runtime, prefer `studio_product_flux_4bit` on Colab, and lower `FLUX_STRENGTH` if it changes product details too aggressively. The 4-bit checkpoint is still large, roughly 13 GB, but it avoids the full 33 GB download. If Hugging Face blocks the full model download, accept the `FLUX.1-schnell` model terms and set `HF_TOKEN` in the runtime.

## Studio Product Generative Mode

`studio_product_generative` is the optional controlled generative pass. It first creates the safer `studio_product` image, then sends that result as a reference image to OpenAI image editing with a strict product-preservation prompt. The prompt asks the model to improve studio lighting, background, camera quality, and shadow while preserving product shape, color, labels, logos, printed text, packaging edges, and item count.

Enable it by setting an API key before starting the server:

```bash
export OPENAI_API_KEY=your_key_here
```

Optional settings:

```bash
export OPENAI_IMAGE_MODEL=gpt-image-2
export OPENAI_IMAGE_SIZE=1024x1024
```

This mode can produce a stronger catalog-photo transformation, but it can still alter small label text or fine packaging details. Keep `studio_product` available as the fidelity-first fallback.

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

It installs the inference stack, downloads `RealESRGAN_x4plus.pth`, lets you upload an image, choose `studio_product_generative`, `studio_product_flux_4bit`, `studio_product_flux`, `studio_product_focus`, `studio_product`, `realesrgan`, or `pillow_fallback`, compares before/after images, and produces an API-style base64 response.

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

- `studio_product_generative`
- `studio_product_flux_4bit`
- `studio_product_flux`
- `studio_product_focus`
- `studio_product`
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
