"""
OpenDoodler AI Image Generation Service
Model: stabilityai/sd-turbo (CPU-safe, auto-detects GPU if available)
"""

import io
import base64
import logging
import time
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from diffusers import AutoPipelineForText2Image
from PIL import Image, ImageDraw, ImageFont
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

pipeline = None
device_info = {}


def load_pipeline():
    global pipeline, device_info

    if torch.cuda.is_available():
        device, dtype, variant = "cuda", torch.float16, "fp16"
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device, dtype, variant = "mps", torch.float16, "fp16"
        log.info("Apple MPS detected")
    else:
        device, dtype, variant = "cpu", torch.float32, None
        log.info("CPU mode — expect ~15–25s per image")

    device_info = {
        "device": device,
        "dtype": str(dtype),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    log.info("Loading stabilityai/sd-turbo…")
    t0 = time.time()

    kwargs = {"torch_dtype": dtype}
    if variant:
        kwargs["variant"] = variant

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", **kwargs)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    global pipeline
    pipeline = pipe
    log.info(f"Pipeline ready in {time.time() - t0:.1f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_pipeline()
    yield
    global pipeline
    pipeline = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="OpenDoodler Image Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = Field(default="", max_length=500)
    steps: int = Field(default=4, ge=1, le=8)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=2.0)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    seed: Optional[int] = Field(default=None)


class GenerateResponse(BaseModel):
    image: str
    width: int
    height: int
    seed: int
    elapsed_ms: int


class AnimationFrameData(BaseModel):
    """Single frame data with SVG or image"""
    frame_data: str  # base64 encoded PNG or SVG as data URI
    duration: float = 0.1  # duration in seconds
    index: int


class ExportMP4Request(BaseModel):
    """Request to export animation as MP4"""
    frames: List[AnimationFrameData]
    fps: int = Field(default=24, ge=1, le=60)
    width: int = Field(default=1024, ge=256, le=4096)
    height: int = Field(default=768, ge=256, le=4096)
    title: Optional[str] = Field(default="OpenDoodler Animation")
    include_audio: bool = False


@app.get("/")
def health():
    return {"status": "ok", "model": "stabilityai/sd-turbo", "pipeline_loaded": pipeline is not None, **device_info}


@app.get("/status")
def status():
    return {"ready": pipeline is not None, "device": device_info.get("device", "unknown")}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")

    seed = req.seed if req.seed is not None else int(time.time() * 1000) % (2 ** 32)
    generator = torch.Generator(device=device_info["device"]).manual_seed(seed)

    log.info(f"Generating '{req.prompt[:60]}' steps={req.steps} {req.width}x{req.height} seed={seed}")
    t0 = time.time()

    try:
        result = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            generator=generator,
        )
    except Exception as e:
        log.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = int((time.time() - t0) * 1000)
    image = result.images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    log.info(f"Done in {elapsed_ms}ms")
    return GenerateResponse(
        image=f"data:image/png;base64,{b64}",
        width=image.width,
        height=image.height,
        seed=seed,
        elapsed_ms=elapsed_ms,
    )


def data_uri_to_image(data_uri: str) -> Image.Image:
    """Convert data URI (PNG/JPG) to PIL Image"""
    try:
        if data_uri.startswith("data:image"):
            # Extract base64 part
            header, data = data_uri.split(",", 1)
            img_data = base64.b64decode(data)
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            raise ValueError("Invalid data URI format")
    except Exception as e:
        log.error(f"Error converting data URI to image: {e}")
        # Return blank image on error
        return Image.new("RGB", (1024, 768), color=(255, 255, 255))


def create_mp4_from_frames(frames: List[Image.Image], fps: int, output_path: str):
    """
    Create MP4 video from PIL Images using OpenCV
    """
    import cv2
    
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].size[1], frames[0].size[0]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert PIL Image to RGB numpy array
        frame_np = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        out.write(frame_np)
    
    out.release()
    log.info(f"MP4 created: {output_path}")


@app.post("/export-mp4")
async def export_mp4(req: ExportMP4Request):
    """
    Export animation frames as MP4 video
    """
    try:
        if not req.frames:
            raise HTTPException(status_code=400, detail="No frames provided")
        
        log.info(f"Exporting {len(req.frames)} frames to MP4 at {req.fps} FPS")
        t0 = time.time()
        
        # Convert data URIs to PIL Images
        images = []
        for frame_data in sorted(req.frames, key=lambda f: f.index):
            img = data_uri_to_image(frame_data.frame_data)
            # Resize to requested dimensions
            if img.size != (req.width, req.height):
                img = img.resize((req.width, req.height), Image.Resampling.LANCZOS)
            images.append(img)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Generate MP4
            create_mp4_from_frames(images, req.fps, output_path)
            
            # Read file and return
            with open(output_path, "rb") as f:
                mp4_data = f.read()
            
            elapsed_ms = int((time.time() - t0) * 1000)
            log.info(f"MP4 export completed in {elapsed_ms}ms, size: {len(mp4_data)} bytes")
            
            return {
                "status": "success",
                "file_size": len(mp4_data),
                "frames": len(images),
                "fps": req.fps,
                "duration_seconds": len(images) / req.fps,
                "elapsed_ms": elapsed_ms,
                "download_url": "/download-mp4"
            }
        finally:
            # Clean up temp file
            if os.path.exists(output_path):
                os.remove(output_path)
    
    except Exception as e:
        log.error(f"MP4 export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Store last generated MP4 in memory for download
_last_mp4_data = None
_last_mp4_filename = None


@app.post("/generate-mp4-file")
async def generate_mp4_file(req: ExportMP4Request):
    """
    Generate and serve MP4 file for download
    """
    global _last_mp4_data, _last_mp4_filename
    
    try:
        if not req.frames:
            raise HTTPException(status_code=400, detail="No frames provided")
        
        log.info(f"Generating MP4 file with {len(req.frames)} frames at {req.fps} FPS")
        t0 = time.time()
        
        # Convert data URIs to PIL Images
        images = []
        for frame_data in sorted(req.frames, key=lambda f: f.index):
            img = data_uri_to_image(frame_data.frame_data)
            # Resize to requested dimensions
            if img.size != (req.width, req.height):
                img = img.resize((req.width, req.height), Image.Resampling.LANCZOS)
            images.append(img)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        
        # Generate MP4
        create_mp4_from_frames(images, req.fps, output_path)
        
        # Read file
        with open(output_path, "rb") as f:
            _last_mp4_data = f.read()
        
        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _last_mp4_filename = f"animation_{timestamp}.mp4"
        
        elapsed_ms = int((time.time() - t0) * 1000)
        log.info(f"MP4 file generated in {elapsed_ms}ms, size: {len(_last_mp4_data)} bytes")
        
        # Clean up temp file
        if os.path.exists(output_path):
            os.remove(output_path)
        
        return {
            "status": "success",
            "file_size": len(_last_mp4_data),
            "frames": len(images),
            "fps": req.fps,
            "duration_seconds": len(images) / req.fps,
            "elapsed_ms": elapsed_ms,
            "filename": _last_mp4_filename
        }
    
    except Exception as e:
        log.error(f"MP4 file generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-mp4")
async def download_mp4():
    """
    Download the last generated MP4 file
    """
    global _last_mp4_data, _last_mp4_filename
    
    if _last_mp4_data is None:
        raise HTTPException(status_code=404, detail="No MP4 file available")
    
    filename = _last_mp4_filename or "animation.mp4"
    
    return StreamingResponse(
        io.BytesIO(_last_mp4_data),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
