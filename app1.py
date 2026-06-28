#!/usr/bin/env python3
"""
NeuroDraw — Optimized AI Sketch-to-Image Generator
==================================================
Changes:
  • Local images now use Flask url_for('static', ...) so they render correctly
  • External Unsplash URLs kept exactly as-is
  • Gallery images use loading="lazy" + onerror fallback
  • ArtStyle selector + API integration preserved
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, abort, jsonify, render_template_string, request, url_for

# ---------------------------------------------------------------------------
# ENVIRONMENT GUARDS
# ---------------------------------------------------------------------------
os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# OPTIONAL ML IMPORTS
# ---------------------------------------------------------------------------
ML_AVAILABLE: bool = False
_torch: Any = None
_diffusers_pipe: Any = None
_diffusers_controlnet: Any = None
_transformers_clip_model: Any = None
_transformers_clip_processor: Any = None

try:
    import torch as _torch_import
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from transformers import CLIPModel, CLIPProcessor

    _torch = _torch_import
    _diffusers_pipe = StableDiffusionControlNetPipeline
    _diffusers_controlnet = ControlNetModel
    _transformers_clip_model = CLIPModel
    _transformers_clip_processor = CLIPProcessor
    ML_AVAILABLE = True
except Exception as _ml_exc:
    print(f"[WARN] ML import failed — running in MOCK mode. Details: {_ml_exc}", file=sys.stderr)

if ML_AVAILABLE:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

class GenerationMode(Enum):
    BASIC = "basic"
    DETAILED = "detailed"
    ARTISTIC = "artistic"


class ArtStyle(Enum):
    PHOTOREALISTIC = "photorealistic"
    ANIME = "anime"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    CONCEPT_ART = "concept_art"
    PIXEL_ART = "pixel_art"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"
    DIGITAL_ART = "digital_art"
    SKETCH = "sketch"
    IMPRESSIONIST = "impressionist"


@dataclass(frozen=True, slots=True)
class AppConfig:
    sd_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_id: str = "lllyasviel/control_v11p_sd15_scribble"
    clip_id: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if ML_AVAILABLE and _torch is not None and _torch.cuda.is_available() else "cpu"
    dtype: Any = field(
        default_factory=lambda: (
            _torch.float16
            if ML_AVAILABLE and _torch is not None and _torch.cuda.is_available()
            else (_torch.float32 if ML_AVAILABLE and _torch is not None else None)
        )
    )
    output_size: Tuple[int, int] = (512, 512)
    max_payload_mb: float = 8.0
    inference_steps: int = 20
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    threaded: bool = True

    @property
    def max_payload_bytes(self) -> int:
        return int(self.max_payload_mb * 1024 * 1024)


# =============================================================================
# LOGGING
# =============================================================================

def _configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("NeuroDraw")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER: Final = _configure_logging()


# =============================================================================
# EXCEPTIONS
# =============================================================================

class NeuroDrawError(Exception):
    pass


class ModelNotReadyError(NeuroDrawError):
    pass


class ValidationError(NeuroDrawError):
    pass


# =============================================================================
# METRICS
# =============================================================================

@contextmanager
def _timed_step(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        LOGGER.debug("Step '%s' completed in %.3fs", name, time.perf_counter() - t0)


# =============================================================================
# MOCK SERVICE
# =============================================================================

class MockGenerationService:
    _PLACEHOLDER_TEXT: Final[str] = "ML Unavailable — Check Console"

    def generate(
        self,
        sketch_data: str,
        prompt: str,
        negative_prompt: str = "",
        mode: GenerationMode = GenerationMode.BASIC,
        art_style: Optional[ArtStyle] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        size = (512, 512)
        img = Image.new("RGB", size, color="#0C0C14")
        draw = ImageDraw.Draw(img)
        for y in range(size[1]):
            r = int(10 + (y / size[1]) * 40)
            g = int(12 + (y / size[1]) * 20)
            b = int(20 + (y / size[1]) * 60)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b))
        try:
            font = ImageFont.truetype("arial.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), self._PLACEHOLDER_TEXT, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((size[0] - tw) // 2, (size[1] - th) // 2), self._PLACEHOLDER_TEXT, fill="#FF5B35", font=font)
        draw.text(
            ((size[0] - tw) // 2, (size[1] - th) // 2 + 40),
            "Install: pip install torch diffusers transformers",
            fill="#EEEEF8",
            font=font,
        )

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        dt = time.perf_counter() - t0

        LOGGER.warning("Mock generation served (ML unavailable). Prompt: %s | Style: %s", prompt, art_style.value if art_style else "none")
        return {
            "success": True,
            "image": f"data:image/png;base64,{b64}",
            "prompt_used": prompt,
            "mock": True,
            "metrics": {
                "total_seconds": round(dt, 2),
                "preprocess_seconds": 0.0,
                "inference_seconds": 0.0,
                "encode_seconds": round(dt, 2),
                "device": "mock",
                "mode": mode.value,
                "art_style": art_style.value if art_style else None,
            },
        }


# =============================================================================
# MODEL MANAGER (Singleton)
# =============================================================================

class ModelManager:
    _instance: Optional[ModelManager] = None
    _inst_lock: threading.Lock = threading.Lock()
    _ready_event: threading.Event = threading.Event()

    def __new__(cls, config: AppConfig) -> ModelManager:
        if cls._instance is None:
            with cls._inst_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: AppConfig) -> None:
        if self._initialized:
            return
        self.cfg = config
        self._lock = threading.RLock()
        self._pipe: Any = None
        self._controlnet: Any = None
        self._clip_model: Any = None
        self._clip_processor: Any = None
        self._load_metrics: Dict[str, float] = {}
        self._initialized = True

    def load(self) -> None:
        if not ML_AVAILABLE:
            LOGGER.warning("ML libraries not available — skipping model load.")
            self._ready_event.set()
            return
        if self._ready_event.is_set():
            return

        with self._lock:
            if self._ready_event.is_set():
                return

            t0 = time.perf_counter()
            LOGGER.info("=== Model Loading Started ===")

            try:
                with _timed_step("controlnet_load"):
                    self._controlnet = _diffusers_controlnet.from_pretrained(
                        self.cfg.controlnet_id,
                        torch_dtype=self.cfg.dtype,
                        use_safetensors=True,
                    )

                with _timed_step("pipeline_load"):
                    self._pipe = _diffusers_pipe.from_pretrained(
                        self.cfg.sd_model_id,
                        controlnet=self._controlnet,
                        torch_dtype=self.cfg.dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                    )

                with _timed_step("optimizations"):
                    self._apply_optimizations()

                with _timed_step("clip_load"):
                    self._clip_model = _transformers_clip_model.from_pretrained(self.cfg.clip_id)
                    self._clip_processor = _transformers_clip_processor.from_pretrained(self.cfg.clip_id)

                self._load_metrics["total_seconds"] = time.perf_counter() - t0
                LOGGER.info(
                    "=== Models Ready (%.2fs) | Device: %s | Dtype: %s ===",
                    self._load_metrics["total_seconds"],
                    self.cfg.device,
                    self.cfg.dtype,
                )
                self._ready_event.set()

            except Exception as exc:
                LOGGER.critical("Model loading failed: %s", exc, exc_info=True)
                raise RuntimeError(f"Model initialization failed: {exc}") from exc

    def _apply_optimizations(self) -> None:
        if self._pipe is None or not ML_AVAILABLE:
            return
        if self.cfg.device == "cuda" and _torch is not None:
            self._pipe = self._pipe.to("cuda")
            if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                    LOGGER.info("Optimization: xFormers enabled")
                except Exception:
                    self._pipe.enable_attention_slicing(1)
                    LOGGER.info("Optimization: attention slicing enabled")
            else:
                self._pipe.enable_attention_slicing(1)
            if hasattr(self._pipe, "enable_vae_slicing"):
                self._pipe.enable_vae_slicing()
            if hasattr(self._pipe, "enable_vae_tiling"):
                self._pipe.enable_vae_tiling()
            if hasattr(_torch, "compile") and _torch.cuda.is_available():
                try:
                    self._pipe.unet = _torch.compile(self._pipe.unet, mode="reduce-overhead", fullgraph=False)
                    LOGGER.info("Optimization: UNet compiled")
                except Exception as exc:
                    LOGGER.warning("torch.compile skipped: %s", exc)
        else:
            LOGGER.warning("CUDA unavailable — CPU inference will be slow.")
            self._pipe.enable_attention_slicing(1)

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def get_pipeline(self) -> Any:
        if not self.is_ready or self._pipe is None:
            raise ModelNotReadyError("AI models are still initializing.")
        return self._pipe

    def get_clip(self) -> Tuple[Any, Any]:
        if not self.is_ready or self._clip_model is None or self._clip_processor is None:
            raise ModelNotReadyError("AI models are still initializing.")
        return self._clip_model, self._clip_processor

    @property
    def metrics(self) -> Dict[str, float]:
        return self._load_metrics.copy()


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

class ImageProcessor:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config

    def decode(self, data_uri: str) -> np.ndarray:
        if not isinstance(data_uri, str) or not data_uri:
            raise ValidationError("Sketch data must be a non-empty string.")
        if len(data_uri) > self.cfg.max_payload_bytes:
            raise ValidationError(f"Sketch payload exceeds {self.cfg.max_payload_mb:.1f} MB limit.")
        try:
            if "," in data_uri:
                header, b64_payload = data_uri.split(",", 1)
                if not header.startswith("data:image/"):
                    raise ValidationError("Invalid data URI: missing 'data:image/' header.")
            else:
                b64_payload = data_uri
            raw = base64.b64decode(b64_payload, validate=True)
            image = Image.open(io.BytesIO(raw))
        except Exception as exc:
            raise ValidationError(f"Invalid image data: {exc}") from exc
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)

    def preprocess(self, image_data: str) -> Image.Image:
        np_img = self.decode(image_data)
        h, w = np_img.shape[:2]
        target_w, target_h = self.cfg.output_size
        if h != target_h or w != target_w:
            np_img = cv2.resize(np_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        if len(np_img.shape) == 3 and np_img.shape[2] == 3:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = np_img
        median = float(np.median(gray))
        lower = int(max(0.0, 0.33 * median))
        upper = int(min(255.0, 1.33 * median))
        edges = cv2.Canny(gray, lower, upper)
        control = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(control)

    @staticmethod
    def encode(image: Image.Image, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=fmt, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{b64}"


# =============================================================================
# GENERATION SERVICE
# =============================================================================

class GenerationService:
    _MODE_PROMPTS: Final[Dict[GenerationMode, str]] = {
        GenerationMode.BASIC: (
            "detailed artwork, masterpiece, professional quality, clean lines, "
            "stunning visuals, best quality"
        ),
        GenerationMode.DETAILED: (
            "hyper-detailed, intricate, 8k resolution, award-winning, "
            "cinematic lighting, perfect composition, photorealistic, "
            "unreal engine 5 render, ray tracing"
        ),
        GenerationMode.ARTISTIC: (
            "artistic masterpiece, oil painting style, vibrant colors, "
            "dramatic lighting, expressive brushstrokes, gallery worthy, "
            "impasto texture, museum quality"
        ),
    }

    _STYLE_PROMPTS: Final[Dict[ArtStyle, str]] = {
        ArtStyle.PHOTOREALISTIC: "photorealistic, 8k uhd, dslr, sharp focus, highly detailed",
        ArtStyle.ANIME: "anime style, studio ghibli, cel shaded, vibrant colors, clean lines",
        ArtStyle.OIL_PAINTING: "oil painting, rich textures, canvas, impasto, classical art",
        ArtStyle.WATERCOLOR: "watercolor painting, soft edges, flowing colors, paper texture",
        ArtStyle.CONCEPT_ART: "concept art, digital painting, artstation, trending, matte painting",
        ArtStyle.PIXEL_ART: "pixel art, 16-bit, retro game style, crisp pixels, dithering",
        ArtStyle.CYBERPUNK: "cyberpunk, neon lights, dystopian, high tech low life, futuristic",
        ArtStyle.FANTASY: "fantasy art, magical, ethereal, lord of the rings, dungeons and dragons",
        ArtStyle.DIGITAL_ART: "digital art, procreate, vibrant, trending on artstation, detailed",
        ArtStyle.SKETCH: "pencil sketch, crosshatching, graphite, hand drawn, monochrome",
        ArtStyle.IMPRESSIONIST: "impressionist, monet style, visible brushstrokes, light study, 19th century",
    }

    def __init__(
        self,
        model_manager: ModelManager,
        image_processor: ImageProcessor,
        config: AppConfig,
    ) -> None:
        self.mm = model_manager
        self.ip = image_processor
        self.cfg = config
        self._inference_lock = threading.Lock()

    def _build_prompt(self, user_prompt: str, mode: GenerationMode, art_style: Optional[ArtStyle] = None) -> str:
        prefix = self._MODE_PROMPTS.get(mode, self._MODE_PROMPTS[GenerationMode.BASIC])
        parts = [prefix]
        if art_style:
            style_prompt = self._STYLE_PROMPTS.get(art_style)
            if style_prompt:
                parts.append(style_prompt)
        parts.append(user_prompt)
        return ", ".join(parts)

    def generate(
        self,
        sketch_data: str,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, signature",
        mode: GenerationMode = GenerationMode.BASIC,
        art_style: Optional[ArtStyle] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.mm.is_ready:
            raise ModelNotReadyError(
                "AI models are still loading in the background. Please retry in a few moments."
            )

        with self._inference_lock:
            t_total = time.perf_counter()
            t_preprocess = time.perf_counter()
            control_image = self.ip.preprocess(sketch_data)
            dt_preprocess = time.perf_counter() - t_preprocess

            final_prompt = self._build_prompt(prompt, mode, art_style)
            steps = num_inference_steps or self.cfg.inference_steps
            scale = guidance_scale or self.cfg.guidance_scale

            pipe = self.mm.get_pipeline()
            t_inference = time.perf_counter()

            try:
                with _torch.inference_mode():
                    result = pipe(
                        prompt=final_prompt,
                        image=control_image,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=scale,
                        controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
                        width=self.cfg.output_size[0],
                        height=self.cfg.output_size[1],
                    )
            except Exception as exc:
                if ML_AVAILABLE and _torch is not None and hasattr(_torch.cuda, "OutOfMemoryError"):
                    try:
                        if isinstance(exc, _torch.cuda.OutOfMemoryError):
                            LOGGER.error("CUDA OOM: %s", exc)
                            _torch.cuda.empty_cache()
                            raise RuntimeError("GPU out of memory. Reduce resolution or enable more optimizations.") from exc
                    except Exception:
                        pass
                LOGGER.error("Inference failure: %s", exc, exc_info=True)
                raise RuntimeError(f"Image generation failed: {exc}") from exc

            dt_inference = time.perf_counter() - t_inference
            generated_image = result.images[0]

            t_encode = time.perf_counter()
            b64_image = self.ip.encode(generated_image)
            dt_encode = time.perf_counter() - t_encode
            dt_total = time.perf_counter() - t_total

            LOGGER.info(
                "Generation OK | total=%.2fs preprocess=%.2fs inference=%.2fs encode=%.2fs mode=%s style=%s",
                dt_total, dt_preprocess, dt_inference, dt_encode, mode.value,
                art_style.value if art_style else "none",
            )

            return {
                "success": True,
                "image": b64_image,
                "prompt_used": final_prompt,
                "metrics": {
                    "total_seconds": round(dt_total, 2),
                    "preprocess_seconds": round(dt_preprocess, 2),
                    "inference_seconds": round(dt_inference, 2),
                    "encode_seconds": round(dt_encode, 2),
                    "device": self.cfg.device,
                    "mode": mode.value,
                    "art_style": art_style.value if art_style else None,
                },
            }


# =============================================================================
# HTML TEMPLATE
# =============================================================================

# NOTE: Put your local Gemini images inside a folder named:
#   static/images/
# so Flask can serve them at /static/images/...
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuroDraw — Turn Sketches Into AI Art</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ──────────────────────────────────────────────
   DESIGN TOKENS
────────────────────────────────────────────── */
:root {
  --void:      #050508;
  --s1:        #0C0C14;
  --s2:        #131320;
  --s3:        #1C1C2E;
  --s4:        #252538;
  --border:    rgba(255,255,255,0.07);
  --border-md: rgba(255,255,255,0.13);
  --ember:     #FF5B35;
  --ember-dim: rgba(255,91,53,0.14);
  --ember-glow:rgba(255,91,53,0.32);
  --lime:      #BAFF57;
  --lime-dim:  rgba(186,255,87,0.12);
  --blue:      #4F9EFF;
  --green:     #34D399;
  --text:      #EEEEF8;
  --t2:        rgba(238,238,248,0.62);
  --t3:        rgba(238,238,248,0.36);
  --t4:        rgba(238,238,248,0.16);
  --font-d: 'Space Grotesk', system-ui, sans-serif;
  --font-b: 'Inter', system-ui, sans-serif;
  --font-m: 'JetBrains Mono', monospace;
  --r:   10px;
  --r-lg:18px;
  --r-xl:28px;
  --ease: cubic-bezier(.22,.68,0,1.2);
}

/* ──────────────────────────────────────────────
   RESET
────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth;-webkit-font-smoothing:antialiased}
body{font-family:var(--font-b);background:var(--void);color:var(--text);overflow-x:hidden;line-height:1.6}
img{max-width:100%;display:block}
a{color:inherit;text-decoration:none}
button{font-family:var(--font-b);cursor:pointer;border:none;background:none}
:focus-visible{outline:2px solid var(--ember);outline-offset:3px}
@media(prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;transition-duration:.01ms!important}}

/* ──────────────────────────────────────────────
   SCROLLBAR
────────────────────────────────────────────── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--void)}
::-webkit-scrollbar-thumb{background:var(--s4);border-radius:3px}

/* ──────────────────────────────────────────────
   NAV
────────────────────────────────────────────── */
.nav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  padding:0 max(24px,calc(50vw - 640px));
  height:64px;display:flex;align-items:center;gap:40px;
  transition:background .3s,backdrop-filter .3s,border-color .3s;
  border-bottom:1px solid transparent;
}
.nav.scrolled{background:rgba(5,5,8,.82);backdrop-filter:blur(16px);border-color:var(--border)}
.nav-logo{display:flex;align-items:center;gap:10px;flex-shrink:0}
.nav-logo-icon{
  width:34px;height:34px;border-radius:9px;background:var(--ember);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.nav-logo-name{font-family:var(--font-d);font-size:18px;font-weight:700;letter-spacing:-.025em}
.nav-links{display:flex;gap:28px;margin-left:auto}
.nav-links a{font-size:14px;color:var(--t2);transition:color .15s}
.nav-links a:hover{color:var(--text)}
.nav-cta{
  padding:0 18px;height:38px;border-radius:var(--r);background:var(--ember);
  font-size:14px;font-weight:600;color:#fff;transition:all .2s;
  display:flex;align-items:center;white-space:nowrap;flex-shrink:0;
}
.nav-cta:hover{box-shadow:0 0 0 3px var(--ember-glow);transform:translateY(-1px)}
.nav-toggle{display:none;width:44px;height:44px;align-items:center;justify-content:center;margin-left:auto}

/* ──────────────────────────────────────────────
   HERO
────────────────────────────────────────────── */
.hero{
  min-height:100svh;padding:112px max(24px,calc(50vw - 640px)) 80px;
  display:flex;align-items:center;gap:56px;
  background:
    radial-gradient(ellipse 60% 50% at 70% 50%,rgba(79,158,255,.06) 0%,transparent 70%),
    radial-gradient(ellipse 40% 60% at 20% 60%,rgba(255,91,53,.05) 0%,transparent 70%),
    var(--void);
  position:relative;overflow:hidden;
}
.hero::after{
  content:'';position:absolute;bottom:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--border-md),transparent);
}
.hero-left{flex:1;min-width:0}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  padding:5px 12px;border-radius:100px;
  background:var(--ember-dim);border:1px solid rgba(255,91,53,.25);
  font-family:var(--font-m);font-size:11px;letter-spacing:.06em;color:var(--ember);
  margin-bottom:28px;
}
.hero-eyebrow-dot{width:6px;height:6px;border-radius:50%;background:var(--ember);animation:pulse-ember 2s ease infinite}
@keyframes pulse-ember{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}
.hero-title{
  font-family:var(--font-d);font-size:clamp(44px,6vw,76px);
  font-weight:800;line-height:1.04;letter-spacing:-.03em;
  color:var(--text);
}
.hero-title em{font-style:normal;color:var(--ember)}
.hero-sub{
  margin-top:22px;font-size:18px;line-height:1.65;color:var(--t2);
  max-width:500px;font-weight:400;
}
.hero-actions{display:flex;gap:12px;margin-top:36px;flex-wrap:wrap}
.btn-primary{
  padding:0 28px;height:52px;border-radius:var(--r);background:var(--ember);
  font-size:15px;font-weight:700;color:#fff;
  display:inline-flex;align-items:center;gap:9px;
  transition:all .2s var(--ease);position:relative;overflow:hidden;
}
.btn-primary::before{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.15),transparent);
  opacity:0;transition:opacity .2s;
}
.btn-primary:hover::before{opacity:1}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 10px 30px var(--ember-glow)}
.btn-ghost{
  padding:0 24px;height:52px;border-radius:var(--r);
  border:1px solid var(--border-md);background:transparent;
  font-size:15px;font-weight:500;color:var(--t2);
  display:inline-flex;align-items:center;gap:8px;
  transition:all .2s;
}
.btn-ghost:hover{border-color:var(--border-md);background:var(--s2);color:var(--text)}
.hero-badges{display:flex;flex-wrap:wrap;gap:8px;margin-top:32px}
.hero-badge{
  padding:5px 12px;border-radius:100px;
  background:var(--s2);border:1px solid var(--border);
  font-family:var(--font-m);font-size:11px;color:var(--t3);
  transition:all .2s;
}
.hero-badge:hover{border-color:var(--border-md);color:var(--t2)}

/* Hero demo widget */
.hero-right{flex-shrink:0;width:min(520px,100%)}
.demo-widget{
  background:var(--s1);border:1px solid var(--border-md);
  border-radius:var(--r-xl);overflow:hidden;
  box-shadow:0 32px 80px rgba(0,0,0,.6),0 0 0 1px var(--border);
}
.demo-topbar{
  height:44px;background:var(--s2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 14px;gap:8px;
}
.demo-dot{width:11px;height:11px;border-radius:50%}
.demo-dot:nth-child(1){background:#FF5F57}
.demo-dot:nth-child(2){background:#FEBC2E}
.demo-dot:nth-child(3){background:#28C840}
.demo-title{margin-left:8px;font-family:var(--font-m);font-size:11px;color:var(--t3)}
.demo-stage{display:grid;grid-template-columns:1fr 1fr;position:relative}
.demo-stage-sketch{
  aspect-ratio:1;background:#fafafa;position:relative;overflow:hidden;
  display:flex;align-items:center;justify-content:center;
}
.demo-label{
  position:absolute;top:10px;left:10px;
  font-family:var(--font-m);font-size:9px;letter-spacing:.06em;
  color:rgba(0,0,0,.35);background:rgba(255,255,255,.8);
  padding:3px 8px;border-radius:100px;
}
.demo-stage-result{
  aspect-ratio:1;position:relative;overflow:hidden;
  background:var(--s3);
}
.demo-divider{
  position:absolute;left:50%;top:0;bottom:0;width:1px;
  background:var(--border-md);z-index:2;
}
.demo-arrow{
  position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
  z-index:3;width:32px;height:32px;border-radius:50%;
  background:var(--s1);border:1px solid var(--border-md);
  display:flex;align-items:center;justify-content:center;
}
.demo-sketch-svg{width:90%;height:90%}
.demo-result-art{
  width:100%;height:100%;position:absolute;inset:0;opacity:0;transition:opacity .8s ease;object-fit:cover;
  image-rendering:-webkit-optimize-contrast;image-rendering:crisp-edges;
}
.demo-result-art.visible{opacity:1}
.demo-result-shimmer{
  position:absolute;inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.04),transparent);
  background-size:200% 100%;
  animation:shimmer 1.2s ease infinite;opacity:0;transition:opacity .3s;
}
.demo-result-shimmer.active{opacity:1}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
.demo-bottom{
  padding:14px 16px;background:var(--s2);border-top:1px solid var(--border);
  display:flex;align-items:center;gap:8px;
}
.demo-gen-btn{
  flex:1;height:38px;border-radius:var(--r);background:var(--ember);
  font-size:13px;font-weight:700;color:#fff;
  display:flex;align-items:center;justify-content:center;gap:7px;
  transition:all .2s;
}
.demo-gen-btn:hover{box-shadow:0 0 0 3px var(--ember-glow)}
.demo-tool-btn{
  width:38px;height:38px;border-radius:var(--r);background:var(--s3);
  border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;color:var(--t3);
  transition:all .15s;
}
.demo-tool-btn:hover{border-color:var(--border-md);color:var(--t2)}

/* ──────────────────────────────────────────────
   MARQUEE
────────────────────────────────────────────── */
.marquee-strip{
  padding:18px 0;background:var(--s1);border-top:1px solid var(--border);
  border-bottom:1px solid var(--border);overflow:hidden;white-space:nowrap;
}
.marquee-inner{
  display:inline-flex;gap:0;
  animation:marquee 28s linear infinite;
}
@keyframes marquee{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.marquee-item{
  display:inline-flex;align-items:center;gap:8px;
  padding:0 28px;font-family:var(--font-m);font-size:12px;color:var(--t3);
  letter-spacing:.04em;
}
.marquee-item::before{content:'';width:5px;height:5px;border-radius:50%;background:var(--border-md);flex-shrink:0}
.marquee-item span{color:var(--t2)}

/* ──────────────────────────────────────────────
   SECTION COMMON
────────────────────────────────────────────── */
.section{padding:96px max(24px,calc(50vw - 640px))}
.section-eyebrow{
  font-family:var(--font-m);font-size:11px;letter-spacing:.08em;
  color:var(--ember);text-transform:uppercase;margin-bottom:14px;
}
.section-title{
  font-family:var(--font-d);font-size:clamp(28px,4vw,46px);
  font-weight:700;line-height:1.1;letter-spacing:-.025em;
  color:var(--text);max-width:600px;
}
.section-sub{margin-top:16px;font-size:17px;color:var(--t2);max-width:520px;line-height:1.65}

/* ──────────────────────────────────────────────
   HOW IT WORKS
────────────────────────────────────────────── */
.how-section{background:var(--void)}
.how-steps{display:grid;grid-template-columns:repeat(3,1fr);gap:0;margin-top:64px;position:relative}
.how-steps::before{
  content:'';position:absolute;top:44px;left:calc(16.6% + 20px);right:calc(16.6% + 20px);
  height:1px;background:linear-gradient(90deg,transparent,var(--border-md),var(--border-md),transparent);
}
.how-step{display:flex;flex-direction:column;align-items:flex-start;padding:0 32px 0 0}
.how-step-num{
  width:88px;height:88px;border-radius:var(--r-lg);
  background:var(--s2);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  position:relative;z-index:1;flex-shrink:0;margin-bottom:28px;
  transition:border-color .3s,background .3s;
}
.how-step:hover .how-step-num{border-color:var(--ember-glow);background:var(--ember-dim)}
.how-step-num svg{width:36px;height:36px;color:var(--t3);transition:color .3s}
.how-step:hover .how-step-num svg{color:var(--ember)}
.how-step-tag{
  font-family:var(--font-m);font-size:10px;letter-spacing:.06em;
  color:var(--t4);margin-bottom:10px;
}
.how-step-title{font-family:var(--font-d);font-size:21px;font-weight:700;color:var(--text);margin-bottom:10px;letter-spacing:-.02em}
.how-step-body{font-size:15px;color:var(--t2);line-height:1.65}

/* ──────────────────────────────────────────────
   FEATURES
────────────────────────────────────────────── */
.features-section{background:var(--s1)}
.features-header{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:56px}
.features-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--border)}
.feat-card{
  background:var(--s1);padding:36px 32px;
  transition:background .25s;
}
.feat-card:hover{background:var(--s2)}
.feat-icon{
  width:52px;height:52px;border-radius:var(--r);
  background:var(--s3);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  margin-bottom:22px;transition:all .25s;
}
.feat-card:hover .feat-icon{background:var(--ember-dim);border-color:rgba(255,91,53,.3)}
.feat-icon svg{width:22px;height:22px;color:var(--t3);transition:color .25s}
.feat-card:hover .feat-icon svg{color:var(--ember)}
.feat-title{font-family:var(--font-d);font-size:17px;font-weight:700;color:var(--text);margin-bottom:10px;letter-spacing:-.02em}
.feat-body{font-size:14px;color:var(--t2);line-height:1.65}

/* ──────────────────────────────────────────────
   INTERACTIVE TRY SECTION
────────────────────────────────────────────── */
.try-section{background:var(--void)}
.try-inner{
  display:grid;grid-template-columns:1fr 1fr;gap:40px;
  margin-top:56px;align-items:start;
}
.try-canvas-wrap{
  background:var(--s1);border:1px solid var(--border-md);
  border-radius:var(--r-xl);overflow:hidden;
}
.try-toolbar{
  height:50px;background:var(--s2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 14px;gap:8px;
}
.try-tool{
  width:38px;height:38px;border-radius:9px;
  background:transparent;border:1px solid transparent;
  display:flex;align-items:center;justify-content:center;
  color:var(--t3);transition:all .15s;
}
.try-tool:hover,.try-tool.active{background:var(--s3);border-color:var(--border);color:var(--text)}
.try-tool.active{background:var(--ember-dim);border-color:rgba(255,91,53,.35);color:var(--ember)}
.try-canvas-cont{position:relative;background:#fff}
#tryCanvas{display:block;cursor:crosshair;touch-action:none}
.try-canvas-hint{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:10px;
  pointer-events:none;transition:opacity .3s;
}
.try-hint-icon{width:48px;height:48px;opacity:.2}
.try-hint-text{font-family:var(--font-m);font-size:13px;color:var(--t3)}
.try-canvas-hint.hidden{opacity:0}
.try-footer{
  padding:14px 16px;background:var(--s2);border-top:1px solid var(--border);
  display:flex;gap:8px;flex-wrap:wrap;
}
.try-style-select{
  height:44px;padding:0 12px;
  background:var(--s3);border:1px solid var(--border);
  border-radius:var(--r);color:var(--text);
  font-size:13px;outline:none;font-family:var(--font-b);
  cursor:pointer;transition:border-color .2s;
  appearance:none;min-width:140px;flex:1;
  background-image:url("data:image/svg+xml,%3Csvg width='10' height='6' viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L5 5L9 1' stroke='%23EEEEF8' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;
  padding-right:32px;
}
.try-style-select:hover,.try-style-select:focus{border-color:var(--border-md)}
.try-generate-btn{
  flex:1;height:44px;border-radius:var(--r);background:var(--ember);
  font-size:14px;font-weight:700;color:#fff;
  display:flex;align-items:center;justify-content:center;gap:8px;
  transition:all .2s;position:relative;overflow:hidden;
  min-width:120px;
}
.try-generate-btn::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.12),transparent);
  opacity:0;transition:opacity .2s;
}
.try-generate-btn:hover::after{opacity:1}
.try-generate-btn:hover{box-shadow:0 0 0 3px var(--ember-glow)}
.try-generate-btn:disabled{background:var(--s3);color:var(--t3);cursor:not-allowed;box-shadow:none}
.try-clear-btn{
  width:44px;height:44px;border-radius:var(--r);
  background:var(--s3);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  color:var(--t2);transition:all .15s;
}
.try-clear-btn:hover{border-color:var(--border-md);color:var(--text)}

/* Result panel */
.try-result-wrap{
  background:var(--s1);border:1px solid var(--border-md);
  border-radius:var(--r-xl);overflow:hidden;
}
.try-result-header{
  height:50px;background:var(--s2);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 16px;gap:8px;
}
.try-result-status{
  width:7px;height:7px;border-radius:50%;background:var(--t4);
  transition:background .3s;
}
.try-result-status.ready{background:var(--green);animation:pulse-green 2s ease infinite}
.try-result-status.working{background:var(--ember);animation:pulse-green 1s ease infinite}
@keyframes pulse-green{0%,100%{opacity:1}50%{opacity:.4}}
.try-result-label{font-family:var(--font-m);font-size:11px;color:var(--t3)}
.try-result-body{aspect-ratio:1;position:relative;overflow:hidden}
.try-result-art{width:100%;height:100%;position:absolute;inset:0;opacity:0;transition:opacity .6s ease}
.try-result-art.visible{opacity:1}
.try-result-empty{
  position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:12px;
  color:var(--t4);transition:opacity .3s;
}
.try-result-empty.hidden{opacity:0;pointer-events:none}
.try-result-empty svg{width:40px;height:40px;opacity:.4}
.try-result-empty p{font-family:var(--font-m);font-size:12px}
.try-shimmer{
  position:absolute;inset:0;background:var(--s3);
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;
  opacity:0;transition:opacity .3s;pointer-events:none;
}
.try-shimmer.active{opacity:1}
.try-shimmer-bar{
  width:60%;height:3px;border-radius:2px;background:var(--s4);overflow:hidden;
}
.try-shimmer-fill{
  height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--ember),var(--blue));
  width:0%;transition:width .4s ease;
}
.try-shimmer-text{font-family:var(--font-m);font-size:11px;color:var(--t3)}

/* ──────────────────────────────────────────────
   GALLERY
────────────────────────────────────────────── */
.gallery-section{background:var(--s1)}
.gallery-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:16px;margin-top:56px;
}
.gallery-card{
  border-radius:var(--r-lg);overflow:hidden;
  border:1px solid var(--border);background:var(--s2);
  transition:all .3s var(--ease);
  content-visibility:auto;
}
.gallery-card:hover{border-color:var(--border-md);transform:translateY(-4px);box-shadow:0 20px 60px rgba(0,0,0,.4)}
.gallery-pair{display:grid;grid-template-columns:1fr 1fr;aspect-ratio:2/1;position:relative}
.gallery-sketch{position:relative;background:#fafafa;display:flex;align-items:center;justify-content:center}
.gallery-result{position:relative;overflow:hidden}.gallery-result img{width:100%;height:100%;object-fit:cover;display:block}
.gallery-split{
  position:absolute;left:50%;top:0;bottom:0;width:1px;
  background:var(--border-md);z-index:2;
}
.gallery-meta{padding:14px 16px}
.gallery-meta-title{font-size:14px;font-weight:600;color:var(--text);margin-bottom:4px}
.gallery-meta-tags{display:flex;flex-wrap:wrap;gap:5px}
.gallery-tag{
  font-family:var(--font-m);font-size:10px;color:var(--t3);
  background:var(--s3);border:1px solid var(--border);
  padding:2px 8px;border-radius:100px;letter-spacing:.03em;
}

/* Sketch SVG art for gallery */
.sketch-art{width:80%;height:80%}

/* ──────────────────────────────────────────────
   STATS
────────────────────────────────────────────── */
.stats-section{
  background:var(--void);
  padding:80px max(24px,calc(50vw - 640px));
}
.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border)}
.stat-card{
  background:var(--void);padding:40px 32px;
  display:flex;flex-direction:column;gap:8px;
}
.stat-num{
  font-family:var(--font-d);font-size:clamp(36px,4vw,52px);
  font-weight:800;letter-spacing:-.03em;
  background:linear-gradient(135deg,var(--text),var(--t2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.stat-num .accent{-webkit-text-fill-color:var(--ember)}
.stat-label{font-size:14px;color:var(--t2)}
.stat-sub{font-family:var(--font-m);font-size:11px;color:var(--t4);margin-top:2px}

/* ──────────────────────────────────────────────
   TESTIMONIALS
────────────────────────────────────────────── */
.testimonials-section{background:var(--s1)}
.testi-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:56px}
.testi-card{
  background:var(--s2);border:1px solid var(--border);border-radius:var(--r-lg);
  padding:28px;display:flex;flex-direction:column;gap:16px;
  transition:all .25s;
}
.testi-card:hover{border-color:var(--border-md);background:var(--s3)}
.testi-stars{display:flex;gap:3px}
.star{width:14px;height:14px;color:var(--ember)}
.testi-quote{font-size:15px;color:var(--t2);line-height:1.7;font-style:italic}
.testi-author{display:flex;align-items:center;gap:12px;margin-top:4px}
.testi-avatar{
  width:38px;height:38px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--font-d);font-size:14px;font-weight:700;color:#fff;
}
.testi-name{font-size:14px;font-weight:600;color:var(--text)}
.testi-role{font-size:12px;color:var(--t3)}

/* ──────────────────────────────────────────────
   CTA
────────────────────────────────────────────── */
.cta-section{
  padding:120px max(24px,calc(50vw - 640px));
  background:var(--void);text-align:center;position:relative;overflow:hidden;
}
.cta-section::before{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 70% 60% at 50% 50%,rgba(255,91,53,.07) 0%,transparent 70%);
}
.cta-inner{position:relative}
.cta-title{
  font-family:var(--font-d);font-size:clamp(36px,5vw,64px);
  font-weight:800;letter-spacing:-.03em;line-height:1.05;margin-bottom:20px;
}
.cta-sub{font-size:18px;color:var(--t2);max-width:480px;margin:0 auto 40px;line-height:1.65}
.cta-actions{display:flex;justify-content:center;gap:12px;flex-wrap:wrap}
.cta-note{margin-top:16px;font-family:var(--font-m);font-size:11px;color:var(--t4)}

/* ──────────────────────────────────────────────
   FOOTER
────────────────────────────────────────────── */
.footer{
  padding:56px max(24px,calc(50vw - 640px)) 36px;
  border-top:1px solid var(--border);background:var(--s1);
}
.footer-top{display:grid;grid-template-columns:260px 1fr 1fr 1fr;gap:40px;margin-bottom:48px}
.footer-brand{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.footer-brand-icon{width:30px;height:30px;border-radius:8px;background:var(--ember);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.footer-brand-name{font-family:var(--font-d);font-size:16px;font-weight:700}
.footer-desc{font-size:14px;color:var(--t3);line-height:1.65;margin-bottom:20px}
.footer-social{display:flex;gap:8px}
.footer-social-btn{
  width:36px;height:36px;border-radius:9px;border:1px solid var(--border);
  background:var(--s2);display:flex;align-items:center;justify-content:center;
  color:var(--t3);transition:all .15s;
}
.footer-social-btn:hover{border-color:var(--border-md);color:var(--text)}
.footer-col-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:16px;letter-spacing:-.01em}
.footer-links{display:flex;flex-direction:column;gap:10px}
.footer-links a{font-size:14px;color:var(--t3);transition:color .15s}
.footer-links a:hover{color:var(--text)}
.footer-bottom{display:flex;align-items:center;justify-content:space-between;gap:16px;padding-top:24px;border-top:1px solid var(--border)}
.footer-copy{font-size:13px;color:var(--t4)}
.footer-legal{display:flex;gap:20px}
.footer-legal a{font-size:13px;color:var(--t4);transition:color .15s}
.footer-legal a:hover{color:var(--t2)}

/* ──────────────────────────────────────────────
   SCROLL REVEAL
────────────────────────────────────────────── */
.reveal{opacity:0;transform:translateY(24px);transition:opacity .6s ease,transform .6s ease}
.reveal.visible{opacity:1;transform:translateY(0)}
.reveal-delay-1{transition-delay:.1s}
.reveal-delay-2{transition-delay:.2s}
.reveal-delay-3{transition-delay:.3s}

/* ──────────────────────────────────────────────
   RESPONSIVE
────────────────────────────────────────────── */
@media(max-width:1024px){
  .features-grid{grid-template-columns:repeat(2,1fr)}
  .gallery-grid{grid-template-columns:repeat(2,1fr)}
  .gallery-card:last-child{display:none}
  .stats-grid{grid-template-columns:repeat(2,1fr)}
  .footer-top{grid-template-columns:1fr 1fr}
  .footer-top > *:first-child{grid-column:1/-1}
  .testi-grid{grid-template-columns:1fr 1fr}
  .testi-grid > *:last-child{display:none}
}
@media(max-width:768px){
  .hero{flex-direction:column;padding-top:96px;gap:40px}
  .hero-right{width:100%}
  .nav-links{display:none}
  .nav-toggle{display:flex}
  .how-steps{grid-template-columns:1fr;gap:36px}
  .how-steps::before{display:none}
  .try-inner{grid-template-columns:1fr}
  .features-grid{grid-template-columns:1fr}
  .gallery-grid{grid-template-columns:1fr}
  .gallery-card:nth-child(n+3){display:none}
  .stats-grid{grid-template-columns:1fr 1fr}
  .testi-grid{grid-template-columns:1fr}
  .testi-grid > *:last-child{display:block}
  .footer-top{grid-template-columns:1fr 1fr}
  .cta-title{font-size:clamp(30px,7vw,50px)}
}
@media(max-width:480px){
  .stats-grid{grid-template-columns:1fr}
  .footer-top{grid-template-columns:1fr}
  .hero-actions{flex-direction:column}
  .btn-primary,.btn-ghost{width:100%;justify-content:center}
  .try-footer{flex-direction:column}
  .try-style-select{width:100%}
}
</style>
<base target="_blank">
</head>
<body>

<!-- NAV -->
<<nav class="nav" id="mainNav" aria-label="Main navigation">
  <a href="#" class="nav-logo" aria-label="NeuroDraw home">
    <div class="nav-logo-icon" aria-hidden="true">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
        <path d="M12 2C8.5 2 6 4.5 6 8c0 2 1 3.5 2.5 4.5L8 20h8l-.5-7.5C17 11.5 18 10 18 8c0-3.5-2.5-6-6-6z" fill="white" opacity=".9"/>
        <path d="M9 20h6M10.5 16.5h3" stroke="rgba(255,91,53,.7)" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </div>
    <span class="nav-logo-name">NeuroDraw</span>
  </a>
  <nav class="nav-links" aria-label="Site sections">
    <a href="#how">How it works</a>
    <a href="#features">Features</a>
    <a href="#gallery">Gallery</a>
    <a href="#try">Try free</a>
  </nav>
  <a href="#try" class="nav-cta" style="margin-left:auto">
    Start Drawing
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>
  </a>
  <button class="nav-toggle" aria-label="Open menu" id="navToggle">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
  </button>
</nav>

<!-- HERO -->
<section class="hero" id="home" aria-label="Hero">
  <div class="hero-left">
    <div class="hero-eyebrow" role="note">
      <span class="hero-eyebrow-dot" aria-hidden="true"></span>
      ControlNet · Stable Diffusion · CLIP
    </div>
    <h1 class="hero-title">
      Your sketch.<br>
      Its <em>vision.</em>
    </h1>
    <p class="hero-sub">
      Draw anything — a rough scribble, a precise line art — and watch NeuroDraw transform it into photorealistic AI artwork in seconds.
    </p>
    <div class="hero-actions">
      <a href="#try" class="btn-primary">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>
          <path d="m15 5 4 4"/>
        </svg>
        Start drawing free
      </a>
      <a href="#how" class="btn-ghost">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="12" cy="12" r="10"/>
          <polygon points="10 8 16 12 10 16 10 8"/>
        </svg>
        See how it works
      </a>
    </div>
    <div class="hero-badges" role="list" aria-label="Technology stack">
      <span class="hero-badge" role="listitem">ControlNet</span>
      <span class="hero-badge" role="listitem">Stable Diffusion</span>
      <span class="hero-badge" role="listitem">CLIP Prompting</span>
      <span class="hero-badge" role="listitem">PyTorch</span>
      <span class="hero-badge" role="listitem">CUDA Accelerated</span>
    </div>
  </div>

  <div class="hero-right" aria-label="Live sketch demo">
    <div class="demo-widget" role="img" aria-label="Animated demo showing sketch transforming to AI art">
      <div class="demo-topbar" aria-hidden="true">
        <div class="demo-dot"></div>
        <div class="demo-dot"></div>
        <div class="demo-dot"></div>
        <span class="demo-title">neurodraw · live demo</span>
      </div>
      <div class="demo-stage">
        <div class="demo-stage-sketch">
          <div class="demo-label">sketch</div>
          <svg class="demo-sketch-svg" viewBox="0 0 200 200" id="demoSvg">
            <path id="ds1" d="M10 160 L70 60 L130 160" stroke="#1a1a2e" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
            <path id="ds2" d="M80 160 L150 55 L220 160" stroke="#1a1a2e" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
            <path id="ds3" d="M0 160 L220 160" stroke="#1a1a2e" stroke-width="2" fill="none" stroke-linecap="round"/>
            <circle id="ds4" cx="160" cy="40" r="18" stroke="#1a1a2e" stroke-width="2" fill="none"/>
            <path id="ds5" d="M30 160 L30 140 L40 120 L50 140 L50 160" stroke="#1a1a2e" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
            <path id="ds6" d="M55 160 L55 145 L63 128 L71 145 L71 160" stroke="#1a1a2e" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
            <path id="ds7" d="M0 170 Q55 165 110 168 Q165 171 220 167" stroke="#1a1a2e" stroke-width="1.5" fill="none" stroke-linecap="round"/>
            <path id="ds8" d="M0 178 Q55 175 110 176 Q165 177 220 174" stroke="#1a1a2e" stroke-width="1.5" fill="none" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="demo-stage-result">
          <div class="demo-label" style="background:rgba(0,0,0,.5);color:rgba(255,255,255,.6)">ai result</div>
          <!-- LOCAL IMAGES: served from Flask static folder -->
          <img class="demo-result-art" id="demoArt1" src="{{ url_for('static', filename='images/Gemini_Generated_Image_a50z0ra50z0ra50z.png') }}" alt="AI generated art 1" fetchpriority="high" decoding="async" onerror="this.style.display='none'">
          <img class="demo-result-art" id="demoArt2" src="{{ url_for('static', filename='images/Gemini_Generated_Image_a50z0ra50z0ra50z.png') }}" alt="AI generated art 2" fetchpriority="high" decoding="async" onerror="this.style.display='none'">
          <img class="demo-result-art" id="demoArt3" src="{{ url_for('static', filename='images/Gemini_Generated_Image_a50z0ra50z0ra50z.png') }}" alt="AI generated art 3" fetchpriority="high" decoding="async" onerror="this.style.display='none'">
          <div class="demo-result-shimmer" id="demoShimmer"></div>
        </div>
        <div class="demo-divider" aria-hidden="true"></div>
        <div class="demo-arrow" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ember)" stroke-width="2.5"><path d="m9 18 6-6-6-6"/></svg>
        </div>
      </div>
      <div class="demo-bottom">
        <button class="demo-tool-btn" aria-label="Pen tool" title="Pen">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
        </button>
        <button class="demo-tool-btn" aria-label="Eraser tool" title="Eraser">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/></svg>
        </button>
        <div class="demo-gen-btn" role="status" aria-live="polite" id="demoBtnStatus">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          <span id="demoStatus">Generating…</span>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- MARQUEE -->
<div class="marquee-strip" aria-hidden="true">
  <div class="marquee-inner" id="marqueeInner">
    <span class="marquee-item"><span>ControlNet</span></span>
    <span class="marquee-item"><span>Stable Diffusion v1.5</span></span>
    <span class="marquee-item"><span>CLIP Encoder</span></span>
    <span class="marquee-item"><span>PyTorch 2.0</span></span>
    <span class="marquee-item"><span>CUDA Acceleration</span></span>
    <span class="marquee-item"><span>scribble ControlNet</span></span>
    <span class="marquee-item"><span>512×512 output</span></span>
    <span class="marquee-item"><span>DDIM Sampler</span></span>
    <span class="marquee-item"><span>Classifier-Free Guidance</span></span>
    <span class="marquee-item"><span>Flask + Python</span></span>
    <span class="marquee-item"><span>ControlNet</span></span>
    <span class="marquee-item"><span>Stable Diffusion v1.5</span></span>
    <span class="marquee-item"><span>CLIP Encoder</span></span>
    <span class="marquee-item"><span>PyTorch 2.0</span></span>
    <span class="marquee-item"><span>CUDA Acceleration</span></span>
    <span class="marquee-item"><span>scribble ControlNet</span></span>
    <span class="marquee-item"><span>512×512 output</span></span>
    <span class="marquee-item"><span>DDIM Sampler</span></span>
    <span class="marquee-item"><span>Classifier-Free Guidance</span></span>
    <span class="marquee-item"><span>Flask + Python</span></span>
  </div>
</div>

<!-- HOW IT WORKS -->
<section class="section how-section" id="how" aria-label="How NeuroDraw works">
  <div class="reveal">
    <p class="section-eyebrow">The process</p>
    <h2 class="section-title">From scribble to masterpiece in three steps</h2>
  </div>
  <div class="how-steps" role="list">
    <div class="how-step reveal reveal-delay-1" role="listitem">
      <div class="how-step-num" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/>
          <path d="m15 5 4 4"/>
        </svg>
      </div>
      <span class="how-step-tag">STEP ONE</span>
      <h3 class="how-step-title">Sketch anything</h3>
      <p class="how-step-body">Open the canvas and draw — loose scribbles, precise line art, or rough shapes. No artistic skill required. The rougher the sketch, the more the AI interprets.</p>
    </div>
    <div class="how-step reveal reveal-delay-2" role="listitem">
      <div class="how-step-num" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          <path d="M12 8v4M12 16h.01"/>
        </svg>
      </div>
      <span class="how-step-tag">STEP TWO</span>
      <h3 class="how-step-title">Describe your vision</h3>
      <p class="how-step-body">Add a text prompt or let CLIP auto-suggest one from your sketch. Choose a style — photorealistic, oil painting, anime, watercolor — and set the generation parameters.</p>
    </div>
    <div class="how-step reveal reveal-delay-3" role="listitem">
      <div class="how-step-num" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
      </div>
      <span class="how-step-tag">STEP THREE</span>
      <h3 class="how-step-title">Generate & export</h3>
      <p class="how-step-body">Hit generate. ControlNet guides Stable Diffusion to follow your exact sketch structure while the AI fills in stunning photorealistic or artistic detail. Download in seconds.</p>
    </div>
  </div>
</section>

<!-- FEATURES -->
<section class="section features-section" id="features" aria-label="NeuroDraw features">
  <div class="features-header">
    <div class="reveal">
      <p class="section-eyebrow">Capabilities</p>
      <h2 class="section-title">Everything a creative studio needs</h2>
    </div>
  </div>
  <div class="features-grid" role="list">
    <div class="feat-card reveal" role="listitem">
      <div class="feat-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/>
          <path d="m9 12 2 2 4-4"/>
        </svg>
      </div>
      <h3 class="feat-title">ControlNet Precision</h3>
      <p class="feat-body">Your sketch is the blueprint. ControlNet ensures the AI respects your exact composition, proportions, and structural intent — not just the prompt.</p>
    </div>
    <div class="feat-card reveal reveal-delay-1" role="listitem">
      <div class="feat-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>
        </svg>
      </div>
      <h3 class="feat-title">CUDA-Accelerated</h3>
      <p class="feat-body">GPU-accelerated inference via CUDA means generations complete in 3–8 seconds on compatible hardware. CPU fallback available for any machine.</p>
    </div>
    <div class="feat-card reveal reveal-delay-2" role="listitem">
      <div class="feat-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <circle cx="9" cy="9" r="2"/>
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
        </svg>
      </div>
      <h3 class="feat-title">30+ Art Styles</h3>
      <p class="feat-body">Photorealistic, anime, oil painting, watercolor, concept art, pixel art, and more. Style tags combine with your prompt for precise creative control.</p>
    </div>
    <div class="feat-card reveal" role="listitem">
      <div class="feat-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <h3 class="feat-title">CLIP Auto-Prompting</h3>
      <p class="feat-body">Not sure what to write? CLIP analyzes your sketch and generates a descriptive prompt automatically, then you can refine it to your taste.</p>
    </div>
    <div class="feat-card reveal reveal-delay-1" role="listitem">
      <div class="feat-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
      </div>
      <h3 class="feat-title">Instant Export</h3>
      <p class="feat-body">Download your generation as PNG, JPG, or WebP at up to 4K resolution. Save your sketch + result pair for reference or share directly from the app.</p>
    </div>
    <div class="feat-card reveal reveal-delay-2" role="listitem">
      <div class="feat-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/>
        </svg>
      </div>
      <h3 class="feat-title">Full Parameter Control</h3>
      <p class="feat-body">Adjust inference steps (10–50), guidance scale, ControlNet conditioning scale, and negative prompts for granular creative control over every generation.</p>
    </div>
  </div>
</section>

<!-- INTERACTIVE TRY SECTION -->
<section class="section try-section" id="try" aria-label="Try NeuroDraw">
  <div class="reveal">
    <p class="section-eyebrow">Try it now</p>
    <h2 class="section-title">Draw something. See what the AI sees.</h2>
    <p class="section-sub">This is the actual NeuroDraw interface. Sketch on the canvas, pick a style, describe your vision, and hit generate.</p>
  </div>
  <div class="try-inner">
    <div class="try-canvas-wrap reveal">
      <div class="try-toolbar" role="toolbar" aria-label="Drawing tools">
        <button class="try-tool active" id="ttPen" onclick="setTryTool('pen')" aria-label="Pen tool" aria-pressed="true" title="Pen (P)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
        </button>
        <button class="try-tool" id="ttBrush" onclick="setTryTool('brush')" aria-label="Brush tool" aria-pressed="false" title="Brush (B)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18.37 2.63 14 7l-1.59-1.59a2 2 0 0 0-2.82 0L8 7l9 9 1.59-1.59a2 2 0 0 0 0-2.82L17 10l4.37-4.37a2.12 2.12 0 1 0-3-3Z"/></svg>
        </button>
        <button class="try-tool" id="ttEraser" onclick="setTryTool('eraser')" aria-label="Eraser tool" aria-pressed="false" title="Eraser (E)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/></svg>
        </button>
        <div style="width:1px;height:24px;background:var(--border);margin:0 4px"></div>
        <label style="display:flex;align-items:center;gap:6px;font-family:var(--font-m);font-size:10px;color:var(--t3);">
          SIZE
          <input type="range" min="2" max="30" value="6" style="-webkit-appearance:none;appearance:none;width:64px;height:3px;border-radius:2px;background:var(--s4);outline:none;cursor:pointer;" id="trySizeSlider" oninput="updateTrySize(this.value)" aria-label="Brush size">
          <span id="trySizeVal" style="color:var(--t2);min-width:20px;">6</span>
        </label>
        <input type="color" value="#000000" id="tryColorPicker" onchange="tryColor=this.value" style="width:28px;height:28px;border-radius:6px;border:1px solid var(--border);cursor:pointer;background:none;padding:0;margin-left:4px" aria-label="Brush color">
        <style>
          #trySizeSlider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:12px;height:12px;border-radius:50%;background:var(--ember);cursor:pointer}
          #trySizeSlider::-moz-range-thumb{width:12px;height:12px;border-radius:50%;background:var(--ember);cursor:pointer;border:none}
        </style>
      </div>
      <div class="try-canvas-cont" style="position:relative;">
        <canvas id="tryCanvas" width="400" height="400" style="width:100%;height:auto;display:block;cursor:crosshair;touch-action:none;" aria-label="Drawing canvas — draw your sketch here"></canvas>
        <div class="try-canvas-hint" id="tryHint" aria-hidden="true">
          <svg class="try-hint-icon" viewBox="0 0 24 24" fill="none" stroke="var(--t3)" stroke-width="1.5"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
          <span class="try-hint-text">Draw something here</span>
        </div>
      </div>
      <div class="try-footer">
        <select class="try-style-select" id="tryStyle" aria-label="Select art style">
          <option value="photorealistic">Photorealistic</option>
          <option value="anime">Anime</option>
          <option value="oil_painting">Oil Painting</option>
          <option value="watercolor">Watercolor</option>
          <option value="concept_art">Concept Art</option>
          <option value="pixel_art">Pixel Art</option>
          <option value="cyberpunk">Cyberpunk</option>
          <option value="fantasy">Fantasy</option>
          <option value="digital_art">Digital Art</option>
          <option value="sketch">Sketch</option>
          <option value="impressionist">Impressionist</option>
        </select>
        <input type="text" id="tryPrompt" placeholder="Describe your vision… e.g. a mystical forest at sunset" style="flex:1;height:44px;padding:0 12px;background:var(--s3);border:1px solid var(--border);border-radius:var(--r);color:var(--text);font-size:13px;outline:none;font-family:var(--font-b);transition:border-color .2s;" onfocus="this.style.borderColor='var(--border-md)'" onblur="this.style.borderColor='var(--border)'" aria-label="Describe your vision">
        <button class="try-generate-btn" id="tryGenBtn" onclick="simulateGenerate()" aria-label="Generate AI image">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          Generate
        </button>
        <button class="try-clear-btn" onclick="clearTryCanvas()" aria-label="Clear canvas" title="Clear canvas">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
        </button>
      </div>
    </div>

    <div class="try-result-wrap reveal reveal-delay-1">
      <div class="try-result-header">
        <div class="try-result-status" id="tryStatus" aria-hidden="true"></div>
        <span class="try-result-label" id="tryStatusLabel">Waiting for sketch</span>
        <span style="margin-left:auto;font-family:var(--font-m);font-size:10px;color:var(--t4)">AI RESULT</span>
      </div>
      <div class="try-result-body">
        <div class="try-result-empty" id="tryEmpty" role="status" aria-live="polite">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/></svg>
          <p>Your artwork appears here</p>
        </div>
        <div class="try-shimmer" id="tryShimmer" aria-label="Generating…" role="status" aria-live="polite">
          <div class="try-shimmer-bar"><div class="try-shimmer-fill" id="tryProgressFill"></div></div>
          <span class="try-shimmer-text" id="tryProgressText">Extracting sketch edges…</span>
        </div>
        <div class="try-result-art" id="tryArt1" style="background:linear-gradient(160deg,#0a0618 0%,#1e0533 20%,#6b21a8 40%,#c026d3 55%,#f97316 70%,#fbbf24 83%,#166534 90%,#14532d 100%)" aria-hidden="true"></div>
        <div class="try-result-art" id="tryArt2" style="background:radial-gradient(ellipse at 50% 0%,#1e3a8a 0%,#0c4a6e 30%,#0ea5e9 50%,#6ee7b7 65%,#d9f99d 80%,#fde68a 90%,#f97316 100%)" aria-hidden="true"></div>
        <div class="try-result-art" id="tryArt3" style="background:radial-gradient(ellipse at 50% 40%,#fde68a 0%,#f97316 20%,#dc2626 40%,#7c3aed 60%,#1e1b4b 80%,#050508 100%)" aria-hidden="true"></div>
        <div class="try-result-art" id="tryArt4" style="background:linear-gradient(135deg,#064e3b 0%,#065f46 20%,#059669 45%,#a7f3d0 65%,#fde68a 80%,#f59e0b 100%)" aria-hidden="true"></div>
        <div class="try-result-art" id="tryArt5" style="background:radial-gradient(circle at 60% 35%,#fffbeb 0%,#fef3c7 12%,#fcd34d 28%,#f97316 50%,#92400e 72%,#1c1917 100%)" aria-hidden="true"></div>
      </div>
      <div style="padding:14px 16px;border-top:1px solid var(--border);background:var(--s2);display:flex;gap:8px;flex-wrap:wrap;">
        <button onclick="downloadTryResult()" style="flex:1;min-width:100px;height:38px;border-radius:var(--r);background:var(--s3);border:1px solid var(--border);color:var(--t2);font-size:13px;font-weight:500;display:flex;align-items:center;justify-content:center;gap:6px;cursor:pointer;transition:all .15s;font-family:var(--font-b);" onmouseover="this.style.borderColor='var(--border-md)';this.style.color='var(--text)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--t2)'" aria-label="Download result">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Download
        </button>
        <button onclick="simulateGenerate()" style="flex:1;min-width:100px;height:38px;border-radius:var(--r);background:var(--s3);border:1px solid var(--border);color:var(--t2);font-size:13px;font-weight:500;display:flex;align-items:center;justify-content:center;gap:6px;cursor:pointer;transition:all .15s;font-family:var(--font-b);" onmouseover="this.style.borderColor='var(--border-md)';this.style.color='var(--text)'" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--t2)'" aria-label="Generate variation">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M3 2v6h6"/><path d="M21 12A9 9 0 0 0 6 5.3L3 8"/><path d="M21 22v-6h-6"/><path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/></svg>
          Variation
        </button>
      </div>
    </div>
  </div>
</section>

<!-- GALLERY -->
<section class="section gallery-section" id="gallery" aria-label="Gallery">
  <div class="reveal">
    <p class="section-eyebrow">Showcase</p>
    <h2 class="section-title">Sketch on the left. Art on the right.</h2>
    <p class="section-sub">Every image was generated from a hand-drawn sketch using ControlNet + Stable Diffusion.</p>
  </div>
  <div class="gallery-grid" role="list">
    <!-- Card 1: Mountain landscape -->
    <div class="gallery-card reveal" role="listitem">
      <div class="gallery-pair">
        <div class="gallery-sketch">
          <svg class="sketch-art" viewBox="0 0 160 160" fill="none" stroke="#1a1a2e" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10 130 L60 50 L110 130" stroke-width="2"/>
            <path d="M70 130 L120 45 L170 130" stroke-width="2"/>
            <path d="M0 130 L170 130" stroke-width="1.5"/>
            <circle cx="130" cy="32" r="14" stroke-width="1.8"/>
            <path d="M22 130 L22 115 L30 100 L38 115 L38 130" stroke-width="1.5"/>
            <path d="M0 140 Q80 136 160 140" stroke-width="1.2"/>
          </svg>
        </div>
        <div class="gallery-result"><img src="{{ url_for('static', filename='images/Gemini_Generated_Image_a50z0ra50z0ra50z.png') }}" alt="Mountain landscape AI art" loading="lazy" decoding="async" onerror="this.style.display='none'"></div>
        <div class="gallery-split" aria-hidden="true"></div>
      </div>
      <div class="gallery-meta">
        <div class="gallery-meta-title">Mountain landscape at dusk</div>
        <div class="gallery-meta-tags">
          <span class="gallery-tag">photorealistic</span>
          <span class="gallery-tag">cinematic</span>
          <span class="gallery-tag">golden hour</span>
        </div>
      </div>
    </div>
    <!-- Card 2: Cat portrait -->
    <div class="gallery-card reveal reveal-delay-1" role="listitem">
      <div class="gallery-pair">
        <div class="gallery-sketch">
          <svg class="sketch-art" viewBox="0 0 160 160" fill="none" stroke="#1a1a2e" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="80" cy="75" r="40" stroke-width="2"/>
            <path d="M55 45 L45 25 L68 38" stroke-width="2"/>
            <path d="M105 45 L115 25 L92 38" stroke-width="2"/>
            <circle cx="68" cy="70" r="7" stroke-width="1.8"/><circle cx="92" cy="70" r="7" stroke-width="1.8"/>
            <path d="M76 83 L80 86 L84 83" stroke-width="1.8"/>
            <path d="M58 90 L48 88M62 93 L50 93M58 96 L48 98" stroke-width="1.5"/>
            <path d="M102 90 L112 88M98 93 L110 93M102 96 L112 98" stroke-width="1.5"/>
            <path d="M55 115 Q80 130 105 115" stroke-width="1.8"/>
          </svg>
        </div>
        <div class="gallery-result"><img src="https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=400&h=200&fit=crop&q=80" alt="Cosmic cat AI art" loading="lazy" decoding="async" onerror="this.style.display='none'"></div>
        <div class="gallery-split" aria-hidden="true"></div>
      </div>
      <div class="gallery-meta">
        <div class="gallery-meta-title">Ethereal cat portrait</div>
        <div class="gallery-meta-tags">
          <span class="gallery-tag">digital art</span>
          <span class="gallery-tag">cosmic</span>
          <span class="gallery-tag">4K</span>
        </div>
      </div>
    </div>
    <!-- Card 3: City skyline -->
    <div class="gallery-card reveal reveal-delay-2" role="listitem">
      <div class="gallery-pair">
        <div class="gallery-sketch">
          <svg class="sketch-art" viewBox="0 0 160 160" fill="none" stroke="#1a1a2e" stroke-linecap="round" stroke-linejoin="round">
            <rect x="10" y="80" width="20" height="60" stroke-width="1.8"/>
            <rect x="35" y="55" width="25" height="85" stroke-width="1.8"/>
            <rect x="65" y="40" width="30" height="100" stroke-width="1.8"/>
            <rect x="100" y="65" width="22" height="75" stroke-width="1.8"/>
            <rect x="127" y="85" width="18" height="55" stroke-width="1.8"/>
            <line x1="0" y1="140" x2="160" y2="140" stroke-width="1.5"/>
            <circle cx="20" cy="25" r="10" stroke-width="1.5"/>
            <path d="M0 140 Q40 137 80 139 Q120 141 160 137" stroke-width="1"/>
            <rect x="17" y="90" width="4" height="4" stroke-width="1"/>
            <rect x="45" y="70" width="4" height="4" stroke-width="1"/>
            <rect x="75" y="55" width="5" height="5" stroke-width="1"/>
          </svg>
        </div>
        <div class="gallery-result"><img src="https://images.unsplash.com/photo-1514565131-fce0801e5785?w=400&h=200&fit=crop&q=80" alt="Cyberpunk city AI art" loading="lazy" decoding="async" onerror="this.style.display='none'"></div>
        <div class="gallery-split" aria-hidden="true"></div>
      </div>
      <div class="gallery-meta">
        <div class="gallery-meta-title">Futuristic city at night</div>
        <div class="gallery-meta-tags">
          <span class="gallery-tag">cyberpunk</span>
          <span class="gallery-tag">neon</span>
          <span class="gallery-tag">cinematic</span>
        </div>
      </div>
    </div>
    <!-- Card 4: Flower -->
    <div class="gallery-card reveal" role="listitem">
      <div class="gallery-pair">
        <div class="gallery-sketch">
          <svg class="sketch-art" viewBox="0 0 160 160" fill="none" stroke="#1a1a2e" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="80" cy="70" r="15" stroke-width="2"/>
            <ellipse cx="80" cy="40" rx="12" ry="18" stroke-width="1.8"/>
            <ellipse cx="80" cy="100" rx="12" ry="18" stroke-width="1.8"/>
            <ellipse cx="50" cy="70" rx="18" ry="12" stroke-width="1.8"/>
            <ellipse cx="110" cy="70" rx="18" ry="12" stroke-width="1.8"/>
            <ellipse cx="59" cy="49" rx="12" ry="18" transform="rotate(45 59 49)" stroke-width="1.8"/>
            <ellipse cx="101" cy="49" rx="12" ry="18" transform="rotate(-45 101 49)" stroke-width="1.8"/>
            <ellipse cx="59" cy="91" rx="12" ry="18" transform="rotate(-45 59 91)" stroke-width="1.8"/>
            <ellipse cx="101" cy="91" rx="12" ry="18" transform="rotate(45 101 91)" stroke-width="1.8"/>
            <path d="M80 85 Q78 110 72 140" stroke-width="1.8"/>
            <path d="M76 115 Q65 110 55 115" stroke-width="1.5"/>
          </svg>
        </div>
        <div class="gallery-result"><img src="{{ url_for('static', filename='images/Gemini_Generated_Image_ff26khff26khff26.png') }}" alt="Bioluminescent flower AI art" loading="lazy" decoding="async" onerror="this.style.display='none'"></div>
        <div class="gallery-split" aria-hidden="true"></div>
      </div>
      <div class="gallery-meta">
        <div class="gallery-meta-title">Bioluminescent bloom</div>
        <div class="gallery-meta-tags">
          <span class="gallery-tag">oil painting</span>
          <span class="gallery-tag">botanical</span>
          <span class="gallery-tag">vibrant</span>
        </div>
      </div>
    </div>
    <!-- Card 5: Dragon -->
    <div class="gallery-card reveal reveal-delay-1" role="listitem">
      <div class="gallery-pair">
        <div class="gallery-sketch">
          <svg class="sketch-art" viewBox="0 0 160 160" fill="none" stroke="#1a1a2e" stroke-linecap="round" stroke-linejoin="round">
            <path d="M30 120 Q50 80 80 60 Q110 40 140 30" stroke-width="2"/>
            <path d="M80 60 Q95 55 100 45 Q105 35 95 30 Q85 35 80 60" stroke-width="1.8"/>
            <path d="M80 60 Q70 75 65 90 Q60 105 70 120 Q80 130 90 120 Q100 105 95 90" stroke-width="1.8"/>
            <path d="M95 90 Q115 85 130 90 Q140 95 135 105" stroke-width="1.5"/>
            <path d="M70 120 Q55 125 45 135 Q35 145 40 150" stroke-width="1.5"/>
            <path d="M90 120 Q95 130 100 145" stroke-width="1.5"/>
            <circle cx="95" cy="33" r="4" stroke-width="1.5"/>
            <path d="M140 30 L148 22 M140 30 L150 28" stroke-width="1.5"/>
          </svg>
        </div>
        <div class="gallery-result"><img src="{{ url_for('static', filename='images/Gemini_Generated_Image_7livki7livki7liv.png') }}" alt="Emerald dragon AI art" loading="lazy" decoding="async" onerror="this.style.display='none'"></div>
        <div class="gallery-split" aria-hidden="true"></div>
      </div>
      <div class="gallery-meta">
        <div class="gallery-meta-title">Emerald serpent dragon</div>
        <div class="gallery-meta-tags">
          <span class="gallery-tag">fantasy</span>
          <span class="gallery-tag">concept art</span>
          <span class="gallery-tag">detailed</span>
        </div>
      </div>
    </div>
    <!-- Card 6: Portrait -->
    <div class="gallery-card reveal reveal-delay-2" role="listitem">
      <div class="gallery-pair">
        <div class="gallery-sketch">
          <svg class="sketch-art" viewBox="0 0 160 160" fill="none" stroke="#1a1a2e" stroke-linecap="round" stroke-linejoin="round">
            <ellipse cx="80" cy="65" rx="35" ry="42" stroke-width="2"/>
            <path d="M45 65 Q40 55 43 45 Q48 32 60 28 Q72 22 80 25 Q100 20 113 32 Q120 42 117 55 L115 65" stroke-width="1.5"/>
            <path d="M63 62 L73 62M87 62 L97 62" stroke-width="1.8"/>
            <path d="M75 75 Q80 80 85 75" stroke-width="1.5"/>
            <path d="M70 88 Q80 95 90 88" stroke-width="1.8"/>
            <path d="M80 107 Q55 115 40 130 Q35 140 38 155" stroke-width="1.8"/>
            <path d="M80 107 Q105 115 120 130 Q125 140 122 155" stroke-width="1.8"/>
            <path d="M45 30 Q50 15 60 12 Q70 8 80 10 Q90 8 100 12 Q110 16 115 30" stroke-width="1.5"/>
          </svg>
        </div>
        <div class="gallery-result"><img src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&h=200&fit=crop&q=80" alt="Golden hour portrait AI art" loading="lazy" decoding="async" onerror="this.style.display='none'"></div>
        <div class="gallery-split" aria-hidden="true"></div>
      </div>
      <div class="gallery-meta">
        <div class="gallery-meta-title">Golden hour portrait</div>
        <div class="gallery-meta-tags">
          <span class="gallery-tag">photorealistic</span>
          <span class="gallery-tag">portrait</span>
          <span class="gallery-tag">warm tones</span>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- STATS -->
<section class="stats-section" aria-label="NeuroDraw statistics">
  <div class="stats-grid">
    <div class="stat-card reveal">
      <div class="stat-num"><span class="accent">3</span>–8<span style="font-size:.5em;-webkit-text-fill-color:var(--t2)">sec</span></div>
      <div class="stat-label">Average generation time on GPU</div>
      <div class="stat-sub">CUDA-accelerated inference</div>
    </div>
    <div class="stat-card reveal reveal-delay-1">
      <div class="stat-num">30<span style="font-size:.5em;-webkit-text-fill-color:var(--t2)">+</span></div>
      <div class="stat-label">Art styles available</div>
      <div class="stat-sub">Photorealistic to abstract</div>
    </div>
    <div class="stat-card reveal reveal-delay-2">
      <div class="stat-num"><span class="accent">512</span><span style="font-size:.4em;-webkit-text-fill-color:var(--t2)">px</span></div>
      <div class="stat-label">Native output resolution</div>
      <div class="stat-sub">Upscale to 4K supported</div>
    </div>
    <div class="stat-card reveal reveal-delay-3">
      <div class="stat-num">100<span style="font-size:.5em;-webkit-text-fill-color:var(--t2)">%</span></div>
      <div class="stat-label">Local & private</div>
      <div class="stat-sub">Runs entirely on your machine</div>
    </div>
  </div>
</section>

<!-- TESTIMONIALS -->
<section class="section testimonials-section" aria-label="User testimonials">
  <div class="reveal">
    <p class="section-eyebrow">From users</p>
    <h2 class="section-title">Designers, artists, and builders love it</h2>
  </div>
  <div class="testi-grid" role="list">
    <div class="testi-card reveal" role="listitem">
      <div class="testi-stars" aria-label="5 out of 5 stars">
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
      </div>
      <p class="testi-quote">"I sketched a rough UI wireframe and NeuroDraw turned it into a photorealistic app mockup in under 10 seconds. It's completely changed how I prototype."</p>
      <div class="testi-author">
        <div class="testi-avatar" style="background:linear-gradient(135deg,#6366f1,#8b5cf6)">A</div>
        <div>
          <div class="testi-name">Arjun Mehta</div>
          <div class="testi-role">Product Designer · Bangalore</div>
        </div>
      </div>
    </div>
    <div class="testi-card reveal reveal-delay-1" role="listitem">
      <div class="testi-stars" aria-label="5 out of 5 stars">
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
      </div>
      <p class="testi-quote">"The ControlNet precision is insane. My rough character sketches come out as polished concept art every time. It's become my go-to tool for game asset ideation."</p>
      <div class="testi-author">
        <div class="testi-avatar" style="background:linear-gradient(135deg,#f97316,#dc2626)">P</div>
        <div>
          <div class="testi-name">Priya Krishnan</div>
          <div class="testi-role">Game Artist · Chennai</div>
        </div>
      </div>
    </div>
    <div class="testi-card reveal reveal-delay-2" role="listitem">
      <div class="testi-stars" aria-label="5 out of 5 stars">
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
        <svg class="star" viewBox="0 0 24 24" fill="currentColor"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
      </div>
      <p class="testi-quote">"I integrated NeuroDraw's Flask API into my creative pipeline. Running locally means no API costs, total privacy, and full control. A ML engineer's dream setup."</p>
      <div class="testi-author">
        <div class="testi-avatar" style="background:linear-gradient(135deg,#0ea5e9,#6366f1)">R</div>
        <div>
          <div class="testi-name">Rohan Verma</div>
          <div class="testi-role">ML Engineer · Hyderabad</div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- CTA SECTION -->
<section class="cta-section" aria-label="Call to action">
  <div class="cta-inner">
    <h2 class="cta-title reveal">
      Stop describing.<br>
      <span style="color:var(--ember)">Start drawing.</span>
    </h2>
    <p class="cta-sub reveal reveal-delay-1">NeuroDraw is free, open-source, and runs entirely on your machine. No cloud, no subscriptions, no limits.</p>
    <div class="cta-actions reveal reveal-delay-2">
      <a href="#try" class="btn-primary" style="font-size:16px;height:56px;padding:0 32px;">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/></svg>
        Open the canvas
      </a>
      <a href="https://github.com" class="btn-ghost" style="font-size:16px;height:56px;padding:0 28px;">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
        View on GitHub
      </a>
    </div>
    <p class="cta-note reveal reveal-delay-3">Open source · MIT License · Runs locally · No account needed</p>
  </div>
</section>

<!-- FOOTER -->
<<footer class="footer" aria-label="Site footer">
  <div class="footer-top">
    <div>
      <div class="footer-brand">
        <div class="footer-brand-icon" aria-hidden="true">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M12 2C8.5 2 6 4.5 6 8c0 2 1 3.5 2.5 4.5L8 20h8l-.5-7.5C17 11.5 18 10 18 8c0-3.5-2.5-6-6-6z" fill="white" opacity=".9"/>
          </svg>
        </div>
        <span class="footer-brand-name">NeuroDraw</span>
      </div>
      <p class="footer-desc">AI-powered sketch-to-image generation using ControlNet and Stable Diffusion. Runs entirely on your local machine.</p>
      <div class="footer-social" aria-label="Social links">
        <a href="#" class="footer-social-btn" aria-label="GitHub">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
        </a>
        <a href="#" class="footer-social-btn" aria-label="Twitter">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
        </a>
        <a href="#" class="footer-social-btn" aria-label="Discord">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M20.317 4.37a19.791 19.791 0 0 0-4.885-1.515.074.074 0 0 0-.079.037c-.21.375-.444.864-.608 1.25a18.27 18.27 0 0 0-5.487 0 12.64 12.64 0 0 0-.617-1.25.077.077 0 0 0-.079-.037A19.736 19.736 0 0 0 3.677 4.37a.07.07 0 0 0-.032.027C.533 9.046-.32 13.58.099 18.057.102 18.079.117 18.1.132 18.11a19.9 19.9 0 0 0 5.993 3.03.078.078 0 0 0 .084-.028c.462-.63.874-1.295 1.226-1.994a.076.076 0 0 0-.041-.106 13.107 13.107 0 0 1-1.872-.892.077.077 0 0 1-.008-.128 10.2 10.2 0 0 0 .372-.292.074.074 0 0 1 .077-.01c3.928 1.793 8.18 1.793 12.062 0a.074.074 0 0 1 .078.01c.12.098.246.198.373.292a.077.077 0 0 1-.006.127 12.299 12.299 0 0 1-1.873.892.077.077 0 0 0-.041.107c.36.698.772 1.362 1.225 1.993a.076.076 0 0 0 .084.028 19.839 19.839 0 0 0 6.002-3.03.077.077 0 0 0 .032-.054c.5-5.177-.838-9.674-3.549-13.66a.061.061 0 0 0-.031-.03z"/></svg>
        </a>
      </div>
    </div>
    <div>
      <div class="footer-col-title">Product</div>
      <nav class="footer-links" aria-label="Product links">
        <a href="#how">How it works</a>
        <a href="#features">Features</a>
        <a href="#gallery">Gallery</a>
        <a href="#try">Try free</a>
        <a href="#">Changelog</a>
      </nav>
    </div>
    <div>
      <div class="footer-col-title">Developers</div>
      <nav class="footer-links" aria-label="Developer links">
        <a href="#">Documentation</a>
        <a href="#">API Reference</a>
        <a href="#">GitHub</a>
        <a href="#">Self-hosting guide</a>
        <a href="#">Model weights</a>
      </nav>
    </div>
    <div>
      <div class="footer-col-title">Resources</div>
      <nav class="footer-links" aria-label="Resource links">
        <a href="#">Blog</a>
        <a href="#">Community</a>
        <a href="#">Examples</a>
        <a href="#">Tutorials</a>
        <a href="#">License</a>
      </nav>
    </div>
  </div>
  <div class="footer-bottom">
    <span class="footer-copy">© 2025 NeuroDraw. Open-source under MIT License.</span>
    <nav class="footer-legal" aria-label="Legal links">
      <a href="#">Privacy</a>
      <a href="#">Terms</a>
      <a href="#">Contact</a>
    </nav>
  </div>
</footer>

<script>
// ──────────────────────────────────────────────
// NAV SCROLL
// ──────────────────────────────────────────────
const mainNav = document.getElementById('mainNav');
window.addEventListener('scroll', () => {
  mainNav.classList.toggle('scrolled', window.scrollY > 40);
}, {passive:true});

// ──────────────────────────────────────────────
// SCROLL REVEAL
// ──────────────────────────────────────────────
const revealEls = document.querySelectorAll('.reveal');
const revealObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if(e.isIntersecting) { e.target.classList.add('visible'); revealObs.unobserve(e.target); }
  });
}, {threshold:.12, rootMargin:'0px 0px -40px 0px'});
revealEls.forEach(el => revealObs.observe(el));

// ──────────────────────────────────────────────
// HERO DEMO ANIMATION
// ──────────────────────────────────────────────
(function heroDemoLoop() {
  const strokes = document.querySelectorAll('#demoSvg path, #demoSvg circle');
  const arts = [document.getElementById('demoArt1'), document.getElementById('demoArt2'), document.getElementById('demoArt3')];
  const shimmer = document.getElementById('demoShimmer');
  const statusEl = document.getElementById('demoStatus');
  let artIdx = 0;

  strokes.forEach(s => {
    const len = s.getTotalLength ? s.getTotalLength() : 100;
    s.style.strokeDasharray = len;
    s.style.strokeDashoffset = len;
    s.style.transition = 'none';
  });

  const phases = ['Drawing sketch…', 'Processing edges…', 'Running ControlNet…', 'Denoising…', 'Generating…', 'Done!'];

  async function runDemo() {
    arts.forEach(a => a.classList.remove('visible'));
    shimmer.style.opacity = '0';
    strokes.forEach(s => { s.style.transition = 'none'; s.style.strokeDashoffset = s.style.strokeDasharray; });
    statusEl.textContent = 'Drawing sketch…';

    await sleep(300);

    for(let i = 0; i < strokes.length; i++) {
      const s = strokes[i];
      s.style.transition = `stroke-dashoffset ${0.5 + i * 0.1}s cubic-bezier(.4,0,.2,1)`;
      s.style.strokeDashoffset = '0';
      await sleep(220);
    }
    await sleep(600);

    statusEl.textContent = 'Processing edges…';
    shimmer.style.opacity = '1';
    shimmer.style.transition = 'opacity .3s';
    for(let i = 1; i < phases.length - 1; i++) {
      await sleep(500);
      statusEl.textContent = phases[i];
    }
    await sleep(400);

    shimmer.style.opacity = '0';
    arts[artIdx % arts.length].classList.add('visible');
    artIdx++;
    statusEl.textContent = 'Done!';
    await sleep(3200);

    runDemo();
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
  setTimeout(runDemo, 800);
})();

// ──────────────────────────────────────────────
// TRY CANVAS
// ──────────────────────────────────────────────
const tryCanvas = document.getElementById('tryCanvas');
const tryCtx = tryCanvas.getContext('2d');
const tryHint = document.getElementById('tryHint');
const tryEmpty = document.getElementById('tryEmpty');
const tryShimmer = document.getElementById('tryShimmer');
const tryProgressFill = document.getElementById('tryProgressFill');
const tryProgressText = document.getElementById('tryProgressText');
const tryStatusDot = document.getElementById('tryStatus');
const tryStatusLabel = document.getElementById('tryStatusLabel');

tryCtx.fillStyle = '#ffffff';
tryCtx.fillRect(0, 0, tryCanvas.width, tryCanvas.height);

let tryIsDrawing = false;
let tryLastX = 0, tryLastY = 0;
let tryTool = 'pen';
let tryColor = '#000000';
let trySize = 6;
let tryHasDrawn = false;
let tryCurrentArt = 0;
const TRY_ARTS = [1,2,3,4,5];

function getTryPos(e) {
  const rect = tryCanvas.getBoundingClientRect();
  const sx = tryCanvas.width / rect.width;
  const sy = tryCanvas.height / rect.height;
  const cx = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const cy = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
  return { x: cx * sx, y: cy * sy };
}

function tryStart(e) {
  e.preventDefault();
  tryIsDrawing = true;
  const p = getTryPos(e);
  tryLastX = p.x; tryLastY = p.y;
  if(!tryHasDrawn) { tryHasDrawn = true; tryHint.classList.add('hidden'); }
}

function tryDraw(e) {
  e.preventDefault();
  if(!tryIsDrawing) return;
  const p = getTryPos(e);
  tryCtx.lineJoin = 'round';
  tryCtx.lineCap = 'round';
  tryCtx.lineWidth = tryTool === 'eraser' ? trySize * 3 : tryTool === 'brush' ? trySize * 1.8 : trySize;
  tryCtx.strokeStyle = tryTool === 'eraser' ? '#ffffff' : tryColor;
  tryCtx.globalAlpha = tryTool === 'brush' ? 0.55 : 1;
  tryCtx.beginPath();
  tryCtx.moveTo(tryLastX, tryLastY);
  tryCtx.lineTo(p.x, p.y);
  tryCtx.stroke();
  tryCtx.globalAlpha = 1;
  tryLastX = p.x; tryLastY = p.y;
}

function tryEnd(e) {
  if(!tryIsDrawing) return;
  tryIsDrawing = false;
  tryCtx.globalAlpha = 1;
}

tryCanvas.addEventListener('mousedown', tryStart);
tryCanvas.addEventListener('mousemove', tryDraw);
tryCanvas.addEventListener('mouseup', tryEnd);
tryCanvas.addEventListener('mouseleave', tryEnd);
tryCanvas.addEventListener('touchstart', tryStart, {passive:false});
tryCanvas.addEventListener('touchmove', tryDraw, {passive:false});
tryCanvas.addEventListener('touchend', tryEnd, {passive:false});

function setTryTool(t) {
  tryTool = t;
  document.querySelectorAll('.try-tool').forEach(b => { b.classList.remove('active'); b.setAttribute('aria-pressed','false'); });
  const ids = {pen:'ttPen',brush:'ttBrush',eraser:'ttEraser'};
  if(ids[t]) { document.getElementById(ids[t]).classList.add('active'); document.getElementById(ids[t]).setAttribute('aria-pressed','true'); }
  tryCanvas.style.cursor = t === 'eraser' ? 'cell' : 'crosshair';
}

function updateTrySize(v) {
  trySize = parseInt(v);
  document.getElementById('trySizeVal').textContent = v;
}

function clearTryCanvas() {
  tryCtx.fillStyle = '#ffffff';
  tryCtx.fillRect(0, 0, tryCanvas.width, tryCanvas.height);
  tryHasDrawn = false;
  tryHint.classList.remove('hidden');
  TRY_ARTS.forEach(i => { const a = document.getElementById('tryArt'+i); if(a) a.classList.remove('visible'); });
  const realArt = document.getElementById('tryArtReal');
  if(realArt) realArt.classList.remove('visible');
  tryEmpty.classList.remove('hidden');
  tryStatusDot.className = 'try-result-status';
  tryStatusLabel.textContent = 'Waiting for sketch';
}

const genPhases = [
  {text:'Extracting sketch edges…', pct:18},
  {text:'Encoding with CLIP…', pct:34},
  {text:'Running ControlNet…', pct:52},
  {text:'Denoising — step 10/20…', pct:65},
  {text:'Denoising — step 18/20…', pct:82},
  {text:'Decoding latent space…', pct:96},
];

async function simulateGenerate() {
  if(!tryHasDrawn) {
    alert('Draw something on the canvas first!');
    return;
  }
  const btn = document.getElementById('tryGenBtn');
  btn.disabled = true;

  tryEmpty.classList.add('hidden');
  TRY_ARTS.forEach(i => { const a = document.getElementById('tryArt'+i); if(a) a.classList.remove('visible'); });
  const realArt = document.getElementById('tryArtReal');
  if(realArt) realArt.classList.remove('visible');
  tryShimmer.classList.add('active');
  tryProgressFill.style.width = '0%';
  tryStatusDot.className = 'try-result-status working';
  tryStatusLabel.textContent = 'Generating…';

  const prompt = document.getElementById('tryPrompt').value || 'detailed artwork, high quality';
  const style = document.getElementById('tryStyle').value;

  try {
    const canvasData = tryCanvas.toDataURL('image/png');
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        sketch: canvasData,
        prompt: prompt,
        mode: 'basic',
        art_style: style,
        num_inference_steps: 20,
        guidance_scale: 7.5
      })
    });
    if(response.ok) {
      const data = await response.json();
      if(data.success && data.image) {
        let artEl = document.getElementById('tryArtReal');
        if(!artEl) {
          artEl = document.createElement('div');
          artEl.id = 'tryArtReal';
          artEl.className = 'try-result-art';
          // FIX: Removed inline opacity:0 so the .visible class can actually show it
          artEl.style.cssText = 'width:100%;height:100%;position:absolute;inset:0;transition:opacity .6s ease;background-size:cover;background-position:center;';
          document.querySelector('.try-result-body').appendChild(artEl);
        }
        artEl.style.backgroundImage = `url(${data.image})`;
        tryShimmer.classList.remove('active');
        artEl.classList.add('visible');
        tryStatusDot.className = 'try-result-status ready';
        tryStatusLabel.textContent = 'Generation complete';
        btn.disabled = false;
        return;
      }
    }
  } catch(err) {
    console.log('API call failed, falling back to demo mode:', err);
  }

  for(const phase of genPhases) {
    tryProgressFill.style.width = phase.pct + '%';
    tryProgressText.textContent = phase.text;
    await sleep(480);
  }
  tryProgressFill.style.width = '100%';
  await sleep(350);

  tryShimmer.classList.remove('active');
  tryCurrentArt = (tryCurrentArt % TRY_ARTS.length) + 1;
  const artEl = document.getElementById('tryArt'+tryCurrentArt);
  if(artEl) artEl.classList.add('visible');

  tryStatusDot.className = 'try-result-status ready';
  tryStatusLabel.textContent = 'Generation complete';
  btn.disabled = false;
}

function downloadTryResult() {
  const realArt = document.getElementById('tryArtReal');
  if(realArt && realArt.classList.contains('visible')) {
    const link = document.createElement('a');
    link.download = 'neurodraw-' + Date.now() + '.png';
    // FIX: Robust extraction of the data-URI from backgroundImage
    const raw = realArt.style.backgroundImage;
    link.href = raw.replace(/^url\(["']?/, '').replace(/["']?\)$/, '');
    link.click();
    return;
  }
  const currentArtEl = document.getElementById('tryArt'+tryCurrentArt);
  if(!currentArtEl || !currentArtEl.classList.contains('visible')) { alert('Generate an image first!'); return; }
  const link = document.createElement('a');
  link.download = 'neurodraw-' + Date.now() + '.png';
  link.href = tryCanvas.toDataURL('image/png');
  link.click();
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

document.addEventListener('keydown', e => {
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
  if(e.key.toLowerCase()==='p') setTryTool('pen');
  if(e.key.toLowerCase()==='b') setTryTool('brush');
  if(e.key.toLowerCase()==='e') setTryTool('eraser');
  if(e.key.toLowerCase()==='g') simulateGenerate();
});

const statNums = document.querySelectorAll('.stat-num');
const statsObs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if(e.isIntersecting) { e.target.style.opacity='1'; statsObs.unobserve(e.target); }
  });
}, {threshold:.3});
statNums.forEach(n => statsObs.observe(n));

const marqueeInner = document.getElementById('marqueeInner');
marqueeInner.addEventListener('mouseenter', () => marqueeInner.style.animationPlayState='paused');
marqueeInner.addEventListener('mouseleave', () => marqueeInner.style.animationPlayState='running');
</script>
</body>
</html>"""


# =============================================================================
# FLASK APPLICATION FACTORY
# =============================================================================

def create_app() -> Flask:
    app = Flask(__name__)
    cfg = AppConfig()
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_payload_bytes

    model_manager = ModelManager(cfg)
    image_processor = ImageProcessor(cfg)

    if ML_AVAILABLE:
        generation_service: Any = GenerationService(model_manager, image_processor, cfg)
        LOGGER.info("Real ML generation service selected.")
    else:
        generation_service = MockGenerationService()
        LOGGER.warning("Mock generation service active — ML dependencies missing.")

    def _load_models_bg() -> None:
        try:
            model_manager.load()
        except Exception:
            LOGGER.exception("Background model loading failed")

    threading.Thread(target=_load_models_bg, daemon=True, name="ModelLoader").start()

    @app.route("/")
    def index() -> str:
        return render_template_string(HTML_TEMPLATE)

    @app.route("/health")
    def health() -> Response:
        return jsonify(
            {
                "status": "healthy",
                "models_loaded": model_manager.is_ready,
                "ml_available": ML_AVAILABLE,
                "device": cfg.device,
                "cuda_available": _torch.cuda.is_available() if ML_AVAILABLE and _torch else False,
                "model_load_seconds": model_manager.metrics.get("total_seconds"),
                "timestamp": time.time(),
            }
        )

    @app.route("/api/status")
    def api_status() -> Response:
        return jsonify(
            {
                "models_loaded": model_manager.is_ready,
                "ml_available": ML_AVAILABLE,
                "cuda_available": _torch.cuda.is_available() if ML_AVAILABLE and _torch else False,
                "device": cfg.device,
            }
        )

    @app.route("/api/styles")
    def api_styles() -> Response:
        return jsonify(
            {
                "styles": [
                    {"value": s.value, "label": s.value.replace("_", " ").title()}
                    for s in ArtStyle
                ],
                "modes": [
                    {"value": m.value, "label": m.value.title()}
                    for m in GenerationMode
                ],
            }
        )

    @app.route("/api/generate", methods=["POST"])
    def api_generate() -> Response:
        if not request.is_json:
            abort(415, description="Content-Type must be application/json")

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Invalid JSON body"}), 400

        sketch = data.get("sketch")
        if not sketch:
            return jsonify({"success": False, "error": "Missing required field: 'sketch'"}), 400

        prompt = data.get("prompt", "detailed artwork, high quality")
        negative_prompt = data.get(
            "negative_prompt", "blurry, low quality, distorted, ugly, bad anatomy, watermark"
        )
        mode_str = data.get("mode", "basic")
        art_style_str = data.get("art_style")

        try:
            mode = GenerationMode(mode_str)
        except ValueError:
            valid = [m.value for m in GenerationMode]
            return jsonify(
                {"success": False, "error": f"Invalid mode '{mode_str}'. Valid: {valid}"}
            ), 400

        art_style: Optional[ArtStyle] = None
        if art_style_str:
            try:
                art_style = ArtStyle(art_style_str)
            except ValueError:
                valid_styles = [s.value for s in ArtStyle]
                return jsonify(
                    {"success": False, "error": f"Invalid art_style '{art_style_str}'. Valid: {valid_styles}"}
                ), 400

        steps = data.get("num_inference_steps", cfg.inference_steps)
        guidance = data.get("guidance_scale", cfg.guidance_scale)

        try:
            result = generation_service.generate(
                sketch_data=sketch,
                prompt=prompt,
                negative_prompt=negative_prompt,
                mode=mode,
                art_style=art_style,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            return jsonify(result)
        except ModelNotReadyError as exc:
            LOGGER.warning("Request rejected: models not ready (%s)", exc)
            return jsonify({"success": False, "error": str(exc)}), 503
        except ValidationError as exc:
            LOGGER.warning("Request rejected: validation error (%s)", exc)
            return jsonify({"success": False, "error": str(exc)}), 400
        except RuntimeError as exc:
            LOGGER.error("Runtime error during generation: %s", exc, exc_info=True)
            return jsonify({"success": False, "error": str(exc)}), 500
        except Exception as exc:
            LOGGER.exception("Unhandled exception in /api/generate")
            return jsonify({"success": False, "error": "Internal server error"}), 500

    @app.errorhandler(413)
    def too_large(_: Any) -> Response:
        return jsonify({"success": False, "error": "Request payload too large"}), 413

    @app.errorhandler(415)
    def unsupported_media_type(e: Any) -> Response:
        return jsonify({"success": False, "error": str(e.description)}), 415

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    return app


if __name__ == "__main__":
    app = create_app()
    LOGGER.info("🚀 NeuroDraw server starting on http://0.0.0.0:5000")
    if not ML_AVAILABLE:
        LOGGER.warning("⚠️  Running in MOCK mode — ML dependencies are missing.")
        LOGGER.warning("   Fix: pip install torch diffusers transformers accelerate safetensors")
    else:
        LOGGER.info("📦 Models loading in background thread...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
