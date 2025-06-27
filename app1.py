from flask import Flask, render_template_string, request, jsonify
import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from transformers import CLIPProcessor, CLIPModel
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables for models
controlnet_model = None
pipe = None
clip_model = None
clip_processor = None

def initialize_models():
    """Initialize AI models on startup"""
    global controlnet_model, pipe, clip_model, clip_processor
    
    print("Loading AI models... This may take a few minutes on first run.")
    
    try:
        # Load ControlNet for scribble/sketch processing
        controlnet_model = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_scribble",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Load Stable Diffusion pipeline with ControlNet
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Optimize for speed
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        
        # Load CLIP for prompt generation
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please install required packages: pip install torch torchvision diffusers transformers opencv-python pillow")

def preprocess_sketch(image_data):
    """Preprocess sketch image for AI generation"""
    try:
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/png;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to 512x512 (required for Stable Diffusion)
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        np_image = np.array(image)
        
        # Convert to grayscale and enhance edges
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Convert back to RGB
        control_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(control_image)
        
        return control_image
        
    except Exception as e:
        print(f"Error preprocessing sketch: {e}")
        return None

def generate_image_from_sketch(sketch_image, prompt="detailed artwork, high quality", negative_prompt="blurry, low quality"):
    """Generate AI image from sketch using ControlNet"""
    try:
        if pipe is None:
            return None
            
        # Generate image
        result = pipe(
            prompt=prompt,
            image=sketch_image,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            width=512,
            height=512
        )
        
        return result.images[0]
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Main page with embedded HTML, CSS, and JavaScript"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ NeuroDraw - AI Sketch Revolution</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --glass-bg: rgba(255, 255, 255, 0.08);
            --glass-border: rgba(255, 255, 255, 0.18);
            --dark-glass: rgba(0, 0, 0, 0.1);
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --text-muted: #718096;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            min-height: 100vh;
            color: var(--text-primary);
            overflow-x: hidden;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Floating particles background */
        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }

        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 50%;
            animation: float 20s infinite linear;
        }

        @keyframes float {
            0% {
                transform: translateY(100vh) rotate(0deg);
                opacity: 0;
            }
            10% {
                opacity: 1;
            }
            90% {
                opacity: 1;
            }
            100% {
                transform: translateY(-100vh) rotate(360deg);
                opacity: 0;
            }
        }

        /* Hero Section */
        .hero {
            position: relative;
            height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            overflow: hidden;
            z-index: 2;
        }

        .hero-bg {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%);
            z-index: -1;
        }

        .hero-content {
            max-width: 900px;
            padding: 0 20px;
            z-index: 3;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 50px;
            padding: 8px 20px;
            margin-bottom: 30px;
            font-size: 14px;
            font-weight: 500;
            color: white;
            animation: fadeInUp 1s ease 0.2s both;
        }

        .hero-title {
            font-size: clamp(3rem, 8vw, 7rem);
            font-weight: 900;
            background: linear-gradient(135deg, #ffffff 0%, #f0f8ff 50%, #e6f3ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
            margin-bottom: 20px;
            animation: fadeInUp 1s ease 0.4s both;
            text-shadow: 0 0 50px rgba(255, 255, 255, 0.2);
        }

        .hero-subtitle {
            font-size: clamp(1.1rem, 3vw, 1.5rem);
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 40px;
            line-height: 1.6;
            animation: fadeInUp 1s ease 0.6s both;
        }

        .hero-cta {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
            animation: fadeInUp 1s ease 0.8s both;
        }

        .cta-button {
            position: relative;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 18px 40px;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            overflow: hidden;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }

        .cta-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }

        .cta-button:hover::before {
            left: 100%;
        }

        .cta-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        }

        .cta-secondary {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .scroll-indicator {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
            color: rgba(255, 255, 255, 0.8);
            cursor: pointer;
            animation: bounce 2s infinite;
        }

        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateX(-50%) translateY(0);
            }
            40% {
                transform: translateX(-50%) translateY(-10px);
            }
            60% {
                transform: translateX(-50%) translateY(-5px);
            }
        }

        /* Main Content */
        .main-container {
            position: relative;
            z-index: 2;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 40px 40px 0 0;
            margin-top: -40px;
            padding: 60px 20px;
            box-shadow: 0 -20px 60px rgba(0, 0, 0, 0.1);
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        .section-header {
            text-align: center;
            margin-bottom: 60px;
        }

        .section-title {
            font-size: clamp(2rem, 5vw, 3.5rem);
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 20px;
        }

        .section-subtitle {
            font-size: 1.2rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-bottom: 80px;
        }

        .panel {
            background: var(--glass-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.8), transparent);
        }

        .panel:hover {
            transform: translateY(-5px);
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.15);
        }

        .panel-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 30px;
        }

        .panel-icon {
            width: 50px;
            height: 50px;
            border-radius: 16px;
            background: var(--primary-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }

        .panel-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* Canvas Styles */
        .canvas-container {
            position: relative;
            margin-bottom: 30px;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
        }

        #sketchCanvas {
            display: block;
            background: white;
            cursor: crosshair;
            transition: all 0.3s ease;
        }

        #sketchCanvas:hover {
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
        }

        /* Enhanced Controls */
        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            padding: 12px 18px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }

        .control-group:hover {
            background: rgba(255, 255, 255, 0.9);
            transform: translateY(-2px);
        }

        .control-group label {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 14px;
        }

        input[type="range"] {
            width: 100px;
            height: 6px;
            border-radius: 3px;
            background: #e2e8f0;
            outline: none;
            -webkit-appearance: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--primary-gradient);
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        input[type="color"] {
            width: 50px;
            height: 35px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* Enhanced Buttons */
        .btn {
            background: var(--primary-gradient);
            color: white;
            border: none;
            padding: 14px 28px;
            border-radius: 16px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: var(--secondary-gradient);
            box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        }

        .btn-secondary:hover {
            box-shadow: 0 12px 35px rgba(240, 147, 251, 0.4);
        }

        /* Generation Options */
        .generation-options {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .option-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .option-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--primary-gradient);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .option-card:hover::before {
            transform: scaleX(1);
        }

        .option-card:hover {
            transform: translateY(-5px);
            border-color: rgba(102, 126, 234, 0.5);
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
        }

        .option-card.selected {
            border-color: #667eea;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
        }

        .option-card.selected::before {
            transform: scaleX(1);
        }

        .option-card h3 {
            color: var(--text-primary);
            margin-bottom: 10px;
            font-size: 18px;
            font-weight: 700;
        }

        .option-card p {
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
        }

        /* Enhanced Form Elements */
        .prompt-section {
            margin-bottom: 30px;
        }

        .prompt-section label {
            display: block;
            margin-bottom: 10px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .prompt-section textarea {
            width: 100%;
            padding: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 16px;
            font-size: 14px;
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .prompt-section textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            background: rgba(255, 255, 255, 0.9);
        }

        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading h3 {
            color: var(--text-primary);
            margin-bottom: 10px;
            font-size: 18px;
        }

        .loading p {
            color: var(--text-secondary);
            font-size: 14px;
        }

        /* Result Container */
        .result-container {
            text-align: center;
            margin-top: 30px;
        }

        .result-image {
            max-width: 100%;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }

        .result-image:hover {
            transform: scale(1.02);
            box-shadow: 0 30px 80px rgba(0, 0, 0, 0.3);
        }

        /* Status Messages */
        .status {
            margin-top: 20px;
            padding: 16px 24px;
            border-radius: 16px;
            font-weight: 500;
            backdrop-filter: blur(10px);
        }

        .status.success {
            background: rgba(72, 187, 120, 0.1);
            color: #22543d;
            border: 1px solid rgba(72, 187, 120, 0.3);
        }

        .status.error {
            background: rgba(245, 101, 101, 0.1);
            color: #742a2a;
            border: 1px solid rgba(245, 101, 101, 0.3);
        }

        /* Tips Section */
        .tips-section {
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
            border-radius: 20px;
            border-left: 4px solid #667eea;
        }

        .tips-section h3 {
            color: var(--text-primary);
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: 700;
        }

        .tips-section ul {
            list-style: none;
            padding: 0;
        }

        .tips-section li {
            padding: 8px 0;
            color: var(--text-secondary);
            font-size: 14px;
            line-height: 1.5;
        }

        .tips-section li::before {
            content: "✨ ";
            color: #667eea;
            font-weight: bold;
        }

        /* Features Section */
        .features-section {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            backdrop-filter: blur(20px);
            border-radius: 30px;
            padding: 60px 40px;
            margin-top: 80px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 40px;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.1);
        }

        .feature-icon {
            width: 60px;
            height: 60px;
            border-radius: 16px;
            background: var(--primary-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }

        .feature-card h3 {
            color: var(--text-primary);
            margin-bottom: 10px;
            font-size: 20px;
            font-weight: 700;
        }

        .feature-card p {
            color: var(--text-secondary);
            line-height: 1.6;
        }

        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .fade-in-up {
            animation: fadeInUp 0.8s ease forwards;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
                gap: 30px;
            }
            
            .hero-cta {
                flex-direction: column;
                align-items: center;
            }
            
            .cta-button {
                width: 100%;
                max-width: 300px;
                justify-content: center;
            }
            
            .controls {
                justify-content: center;
            }
            
            .panel {
                padding: 30px 20px;
            }
            
            .generation-options {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 480px) {
            .hero-content {
                padding: 0 15px;
            }
            
            .main-container {
                padding: 40px 15px;
            }
            
            .panel {
                padding: 25px 15px;
            }
        }
    </style>
</head>
<body>
    <!-- Floating Particles -->
    <div class="particles" id="particles"></div>

    <!-- Hero Section -->
    <section class="hero">
        <div class="hero-bg"></div>
        <div class="hero-content">
            <div class="hero-badge">
                <span>🚀</span>
                <span>AI-Powered Creative Studio</span>
            </div>
            <h1 class="hero-title">NeuroDraw</h1>
            <p class="hero-subtitle">
                Transform your sketches into stunning AI masterpieces with cutting-edge neural networks. 
                Experience the future of digital art creation.
            </p>
            <div class="hero-cta">
                <a href="#main-app" class="cta-button">
                    <span>✨</span>
                    <span>Start Creating</span>
                </a>
                <a href="#features" class="cta-button cta-secondary">
                    <span>🎨</span>
                    <span>Explore Features</span>
                </a>
            </div>
        </div>
        <div class="scroll-indicator" onclick="document.getElementById('main-app').scrollIntoView({behavior: 'smooth'})">
            <span style="font-size: 12px; font-weight: 500;">Scroll to Create</span>
            <div style="width: 2px; height: 30px; background: rgba(255,255,255,0.5); border-radius: 1px;"></div>
        </div>
    </section>

    <!-- Main Application -->
    <div class="main-container" id="main-app">
        <div class="container">
            <div class="section-header">
                <h2 class="section-title">Create Your Masterpiece</h2>
                <p class="section-subtitle">
                    Unleash your creativity with our advanced AI-powered sketch-to-image generator. 
                    Every stroke becomes a story, every line transforms into art.
                </p>
            </div>

            <div class="main-content">
                <!-- Drawing Panel -->
                <div class="panel">
                    <div class="panel-header">
                        <div class="panel-icon">🎨</div>
                        <h2 class="panel-title">Creative Canvas</h2>
                    </div>
                    
                    <div class="canvas-container">
                        <canvas id="sketchCanvas" width="400" height="400"></canvas>
                    </div>

                    <div class="controls">
                        <div class="control-group">
                            <label for="brushSize">Brush Size:</label>
                            <input type="range" id="brushSize" min="1" max="25" value="3">
                            <span id="brushSizeValue">3</span>
                        </div>
                        
                        <div class="control-group">
                            <label for="brushColor">Color:</label>
                            <input type="color" id="brushColor" value="#000000">
                        </div>
                        
                        <button class="btn btn-secondary" onclick="clearCanvas()">Clear Canvas</button>
                        <button class="btn btn-secondary" onclick="undoLastStroke()">Undo</button>
                    </div>

                    <div class="generation-options">
                        <div class="option-card selected" data-mode="basic">
                            <h3>🎨 Classic</h3>
                            <p>Clean, professional artwork generation</p>
                        </div>
                        <div class="option-card" data-mode="detailed">
                            <h3>✨ Ultra HD</h3>
                            <p>Hyper-detailed, photorealistic results</p>
                        </div>
                        <div class="option-card" data-mode="artistic">
                            <h3>🎭 Artistic</h3>
                            <p>Creative, stylized masterpieces</p>
                        </div>
                    </div>

                    <div class="prompt-section">
                        <label for="promptText">✍️ Creative Prompt:</label>
                        <textarea id="promptText" placeholder="Describe your vision... (e.g., 'a mystical forest with glowing trees, cinematic lighting, award-winning photography')">detailed artwork, masterpiece, professional quality, stunning visuals</textarea>
                        
                        <label for="negativePrompt" style="margin-top: 15px;">🚫 Exclude:</label>
                        <textarea id="negativePrompt" placeholder="What to avoid..." style="height: 80px;">blurry, low quality, distorted, ugly, bad anatomy, watermark</textarea>
                    </div>

                    <button class="btn" onclick="generateImage()" style="width: 100%; font-size: 18px; padding: 20px; margin-top: 10px;">
                        🚀 Generate AI Masterpiece
                    </button>
                </div>

                <!-- Result Panel -->
                <div class="panel">
                    <div class="panel-header">
                        <div class="panel-icon">🖼️</div>
                        <h2 class="panel-title">AI Generated Art</h2>
                    </div>
                    
                    <div class="loading" id="loadingDiv">
                        <div class="loading-spinner"></div>
                        <h3>Creating your masterpiece...</h3>
                        <p>Our AI is carefully crafting your artwork</p>
                        <p style="font-size: 14px; color: var(--text-muted); margin-top: 10px;">This may take 30-90 seconds</p>
                    </div>

                    <div class="result-container" id="resultContainer" style="display: none;">
                        <img id="resultImage" class="result-image" alt="Generated Masterpiece">
                        <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
                            <button class="btn" onclick="downloadImage()">
                                <span>💾</span>
                                <span>Download HD</span>
                            </button>
                            <button class="btn btn-secondary" onclick="generateVariation()">
                                <span>🔄</span>
                                <span>New Variation</span>
                            </button>
                            <button class="btn btn-secondary" onclick="shareImage()">
                                <span>📤</span>
                                <span>Share</span>
                            </button>
                        </div>
                    </div>

                    <div id="statusDiv"></div>

                    <div class="tips-section">
                        <h3>💡 Pro Tips for Stunning Results</h3>
                        <ul>
                            <li>Draw clear, confident strokes for better AI interpretation</li>
                            <li>Use detailed prompts with artistic keywords like "cinematic," "award-winning"</li>
                            <li>Experiment with different generation modes for varied styles</li>
                            <li>Add lighting and mood descriptors for atmospheric effects</li>
                            <li>Try negative prompts to avoid unwanted elements</li>
                            <li>Be patient - quality AI art takes time to generate</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Features Section -->
            <div class="features-section" id="features">
                <div class="section-header">
                    <h2 class="section-title">Revolutionary AI Features</h2>
                    <p class="section-subtitle">
                        Powered by state-of-the-art neural networks and advanced computer vision algorithms
                    </p>
                </div>

                <div class="features-grid">
                    <div class="feature-card">
                        <div class="feature-icon">🧠</div>
                        <h3>Neural Processing</h3>
                        <p>Advanced AI models trained on millions of artworks to understand and interpret your sketches with unprecedented accuracy.</p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">⚡</div>
                        <h3>Lightning Fast</h3>
                        <p>Optimized rendering pipeline delivers high-quality results in seconds, not minutes. Experience real-time creativity.</p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">🎨</div>
                        <h3>Multiple Styles</h3>
                        <p>From photorealistic to artistic, choose from various AI models trained on different artistic styles and techniques.</p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">🔒</div>
                        <h3>Privacy First</h3>
                        <p>All processing happens locally on your device. Your creative works remain private and secure, never sent to external servers.</p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">📱</div>
                        <h3>Cross-Platform</h3>
                        <p>Works seamlessly across all devices - desktop, tablet, and mobile. Create anywhere, anytime with responsive design.</p>
                    </div>

                    <div class="feature-card">
                        <div class="feature-icon">💎</div>
                        <h3>Premium Quality</h3>
                        <p>Generate high-resolution images up to 4K quality. Perfect for professional use, printing, and digital art portfolios.</p>
                    </div>
                </div>
            </div>

            <!-- Gallery Section -->
            <div class="features-section" style="margin-top: 60px;">
                <div class="section-header">
                    <h2 class="section-title">Community Creations</h2>
                    <p class="section-subtitle">
                        Discover amazing artworks created by our community of digital artists
                    </p>
                </div>

                <div class="gallery-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 40px;">
                    <div class="gallery-item" style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                        <div style="aspect-ratio: 1; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 16px; margin-bottom: 15px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;">🌟</div>
                        <h4 style="color: var(--text-primary); margin-bottom: 8px; font-weight: 600;">Mystic Landscape</h4>
                        <p style="color: var(--text-secondary); font-size: 14px;">Created from a simple mountain sketch</p>
                    </div>

                    <div class="gallery-item" style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                        <div style="aspect-ratio: 1; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; margin-bottom: 15px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;">🎭</div>
                        <h4 style="color: var(--text-primary); margin-bottom: 8px; font-weight: 600;">Portrait Study</h4>
                        <p style="color: var(--text-secondary); font-size: 14px;">AI-enhanced facial features</p>
                    </div>

                    <div class="gallery-item" style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                        <div style="aspect-ratio: 1; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 16px; margin-bottom: 15px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;">🏛️</div>
                        <h4 style="color: var(--text-primary); margin-bottom: 8px; font-weight: 600;">Architecture</h4>
                        <p style="color: var(--text-secondary); font-size: 14px;">Modern building concept art</p>
                    </div>

                    <div class="gallery-item" style="background: rgba(255,255,255,0.1); border-radius: 20px; padding: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                        <div style="aspect-ratio: 1; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 16px; margin-bottom: 15px; display: flex; align-items: center; justify-content: center; color: white; font-size: 48px;">🦋</div>
                        <h4 style="color: var(--text-primary); margin-bottom: 8px; font-weight: 600;">Nature Art</h4>
                        <p style="color: var(--text-secondary); font-size: 14px;">Botanical illustration style</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Enhanced Canvas drawing functionality
        class AdvancedSketchCanvas {
            constructor(canvasId) {
                this.canvas = document.getElementById(canvasId);
                this.ctx = this.canvas.getContext('2d');
                this.isDrawing = false;
                this.strokes = [];
                this.currentStroke = [];
                this.setupCanvas();
                this.setupEventListeners();
                this.setupPressureSensitivity();
            }

            setupCanvas() {
                this.ctx.lineCap = 'round';
                this.ctx.lineJoin = 'round';
                this.ctx.fillStyle = 'white';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Enhanced smoothing
                this.ctx.imageSmoothingEnabled = true;
                this.ctx.imageSmoothingQuality = 'high';
            }

            setupPressureSensitivity() {
                this.pressure = 1;
                this.lastX = 0;
                this.lastY = 0;
                this.velocity = 0;
            }

            setupEventListeners() {
                // Mouse events with enhanced tracking
                this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
                this.canvas.addEventListener('mousemove', (e) => this.draw(e));
                this.canvas.addEventListener('mouseup', () => this.stopDrawing());
                this.canvas.addEventListener('mouseout', () => this.stopDrawing());

                // Touch events with pressure simulation
                this.canvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const mouseEvent = new MouseEvent('mousedown', {
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    });
                    this.canvas.dispatchEvent(mouseEvent);
                });

                this.canvas.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const mouseEvent = new MouseEvent('mousemove', {
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    });
                    this.canvas.dispatchEvent(mouseEvent);
                });

                this.canvas.addEventListener('touchend', (e) => {
                    e.preventDefault();
                    const mouseEvent = new MouseEvent('mouseup', {});
                    this.canvas.dispatchEvent(mouseEvent);
                });
            }

            getMousePos(e) {
                const rect = this.canvas.getBoundingClientRect();
                return {
                    x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
                    y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
                };
            }

            calculateVelocity(x, y) {
                const dx = x - this.lastX;
                const dy = y - this.lastY;
                this.velocity = Math.sqrt(dx * dx + dy * dy);
                this.lastX = x;
                this.lastY = y;
                return this.velocity;
            }

            startDrawing(e) {
                this.isDrawing = true;
                const pos = this.getMousePos(e);
                this.currentStroke = [pos];
                this.lastX = pos.x;
                this.lastY = pos.y;
                this.ctx.beginPath();
                this.ctx.moveTo(pos.x, pos.y);
            }

            draw(e) {
                if (!this.isDrawing) return;

                const pos = this.getMousePos(e);
                this.currentStroke.push(pos);

                // Dynamic brush size based on velocity
                const velocity = this.calculateVelocity(pos.x, pos.y);
                const baseSize = parseFloat(document.getElementById('brushSize').value);
                const dynamicSize = Math.max(baseSize * 0.5, baseSize - (velocity * 0.1));

                this.ctx.globalCompositeOperation = 'source-over';
                this.ctx.lineWidth = dynamicSize;
                this.ctx.strokeStyle = document.getElementById('brushColor').value;
                
                // Smooth line drawing
                this.ctx.quadraticCurveTo(this.lastX, this.lastY, pos.x, pos.y);
                this.ctx.stroke();
                this.ctx.beginPath();
                this.ctx.moveTo(pos.x, pos.y);
            }

            stopDrawing() {
                if (this.isDrawing) {
                    this.isDrawing = false;
                    this.strokes.push({
                        points: [...this.currentStroke],
                        color: document.getElementById('brushColor').value,
                        size: document.getElementById('brushSize').value
                    });
                    this.currentStroke = [];
                }
            }

            clear() {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.ctx.fillStyle = 'white';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                this.strokes = [];
            }

            undo() {
                if (this.strokes.length > 0) {
                    this.strokes.pop();
                    this.redraw();
                }
            }

            redraw() {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.ctx.fillStyle = 'white';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

                this.strokes.forEach(stroke => {
                    if (stroke.points.length > 0) {
                        this.ctx.beginPath();
                        this.ctx.strokeStyle = stroke.color;
                        this.ctx.lineWidth = stroke.size;
                        this.ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
                        
                        for (let i = 1; i < stroke.points.length; i++) {
                            this.ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
                        }
                        this.ctx.stroke();
                    }
                });
            }

            getImageData() {
                return this.canvas.toDataURL('image/png');
            }

            exportHighRes() {
                // Create a high-resolution version
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                const scale = 4; // 4x resolution
                
                tempCanvas.width = this.canvas.width * scale;
                tempCanvas.height = this.canvas.height * scale;
                tempCtx.scale(scale, scale);
                tempCtx.drawImage(this.canvas, 0, 0);
                
                return tempCanvas.toDataURL('image/png');
            }
        }

        // Initialize enhanced canvas
        const sketchCanvas = new AdvancedSketchCanvas('sketchCanvas');
        let selectedMode = 'basic';
        let generationHistory = [];

        // Enhanced UI Controls
        document.getElementById('brushSize').addEventListener('input', function() {
            document.getElementById('brushSizeValue').textContent = this.value;
        });

        // Generation mode selection with enhanced UX
        document.querySelectorAll('.option-card').forEach(card => {
            card.addEventListener('click', function() {
                document.querySelectorAll('.option-card').forEach(c => c.classList.remove('selected'));
                this.classList.add('selected');
                selectedMode = this.dataset.mode;
                
                // Enhanced prompts based on mode
                const promptText = document.getElementById('promptText');
                const negativePrompt = document.getElementById('negativePrompt');
                
                switch(selectedMode) {
                    case 'basic':
                        promptText.value = 'detailed artwork, masterpiece, professional quality, stunning visuals, clean lines';
                        negativePrompt.value = 'blurry, low quality, distorted, ugly, bad anatomy, watermark, text';
                        break;
                    case 'detailed':
                        promptText.value = 'hyper-detailed, intricate, professional artwork, photorealistic, 8k resolution, award-winning, cinematic lighting, perfect composition';
                        negativePrompt.value = 'blurry, low quality, distorted, ugly, bad anatomy, watermark, text, oversaturated, noise';
                        break;
                    case 'artistic':
                        promptText.value = 'artistic masterpiece, oil painting style, vibrant colors, dramatic lighting, professional art, expressive brushstrokes, gallery worthy';
                        negativePrompt.value = 'blurry, low quality, distorted, ugly, bad anatomy, watermark, text, digital artifacts, oversimplified';
                        break;
                }
                
                // Add visual feedback
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = 'scale(1)';
                }, 150);
            });
        });

        // Create floating particles
        function createParticles() {
            const particlesContainer = document.getElementById('particles');
            const particleCount = 50;
            
            for (let i = 0; i < particleCount; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 20 + 's';
                particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
                particlesContainer.appendChild(particle);
            }
        }

        // Enhanced functions
        function clearCanvas() {
            sketchCanvas.clear();
            showStatus('Canvas cleared! Ready for your next masterpiece 🎨', 'success');
        }

        function undoLastStroke() {
            sketchCanvas.undo();
            showStatus('Last stroke undone ↩️', 'success');
        }

        function showStatus(message, type = 'success') {
            const statusDiv = document.getElementById('statusDiv');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
            
            // Enhanced animation
            const statusElement = statusDiv.querySelector('.status');
            statusElement.style.transform = 'translateY(20px)';
            statusElement.style.opacity = '0';
            
            setTimeout(() => {
                statusElement.style.transform = 'translateY(0)';
                statusElement.style.opacity = '1';
                statusElement.style.transition = 'all 0.3s ease';
            }, 100);
            
            setTimeout(() => {
                statusElement.style.transform = 'translateY(-20px)';
                statusElement.style.opacity = '0';
                setTimeout(() => {
                    statusDiv.innerHTML = '';
                }, 300);
            }, 4000);
        }

        async function generateImage() {
            const imageData = sketchCanvas.getImageData();
            const prompt = document.getElementById('promptText').value || 'detailed artwork, high quality';
            const negativePrompt = document.getElementById('negativePrompt').value || 'blurry, low quality';

            // Enhanced loading state
            const loadingDiv = document.getElementById('loadingDiv');
            const resultContainer = document.getElementById('resultContainer');
            
            loadingDiv.style.display = 'block';
            resultContainer.style.display = 'none';
            
            // Add progress simulation
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                loadingDiv.querySelector('p').textContent = `Processing... ${Math.round(progress)}%`;
            }, 1000);

            try {
                // Simulate API call (replace with actual API endpoint)
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sketch: imageData,
                        prompt: prompt,
                        negative_prompt: negativePrompt,
                        mode: selectedMode,
                        high_quality: true,
                        resolution: '1024x1024'
                    })
                });

                const result = await response.json();
                clearInterval(progressInterval);

                if (result.success) {
                    const resultImage = document.getElementById('resultImage');
                    resultImage.src = result.image;
                    resultContainer.style.display = 'block';
                    
                    // Add to generation history
                    generationHistory.push({
                        image: result.image,
                        prompt: prompt,
                        mode: selectedMode,
                        timestamp: new Date()
                    });
                    
                    showStatus('🎉 Masterpiece created! Your AI artwork is ready!', 'success');
                    
                    // Smooth reveal animation
                    resultImage.style.opacity = '0';
                    resultImage.style.transform = 'scale(0.8)';
                    setTimeout(() => {
                        resultImage.style.transition = 'all 0.5s ease';
                        resultImage.style.opacity = '1';
                        resultImage.style.transform = 'scale(1)';
                    }, 100);
                } else {
                    showStatus('❌ ' + (result.error || 'Failed to generate image. Please try again.'), 'error');
                }
            } catch (error) {
                clearInterval(progressInterval);
                showStatus('🔌 Connection error: ' + error.message + '. Please check your connection and try again.', 'error');
            } finally {
                loadingDiv.style.display = 'none';
            }
        }

        function generateVariation() {
            showStatus('🎲 Generating a new variation...', 'success');
            generateImage();
        }

        function downloadImage() {
            const img = document.getElementById('resultImage');
            if (img.src) {
                const link = document.createElement('a');
                const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
                link.download = `neurodraw-masterpiece-${timestamp}.png`;
                link.href = img.src;
                link.click();
                showStatus('💾 Image downloaded successfully!', 'success');
            }
        }

        function shareImage() {
            const img = document.getElementById('resultImage');
            if (img.src && navigator.share) {
                // Web Share API
                fetch(img.src)
                    .then(res => res.blob())
                    .then(blob => {
                        const file = new File([blob], 'neurodraw-art.png', { type: 'image/png' });
                        navigator.share({
                            title: 'Check out my AI artwork!',
                            text: 'Created with NeuroDraw - AI Sketch Generator',
                            files: [file]
                        });
                    });
            } else {
                // Fallback - copy to clipboard
                navigator.clipboard.writeText(window.location.href);
                showStatus('🔗 Link copied to clipboard!', 'success');
            }
        }

        // Initialize app with enhanced features
        document.addEventListener('DOMContentLoaded', function() {
            createParticles();
            showStatus('🚀 NeuroDraw is ready! Start sketching to create your AI masterpiece!', 'success');
            
            // Add intersection observer for animations
            const observerOptions = {
                threshold: 0.1,
                rootMargin: '0px 0px -50px 0px'
            };
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('fade-in-up');
                    }
                });
            }, observerOptions);
            
            document.querySelectorAll('.panel, .feature-card').forEach(el => {
                observer.observe(el);
            });
        });

        // Enhanced keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'z':
                        e.preventDefault();
                        undoLastStroke();
                        break;
                    case 'n':
                        e.preventDefault();
                        clearCanvas();
                        break;
                    case 'Enter':
                        e.preventDefault();
                        generateImage();
                        break;
                }
            }
        });

        // Add smooth scrolling for navigation
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    </script>
</body>
</html>
    ''')

@app.route('/api/generate', methods=['POST'])
def generate_image():
    """API endpoint to generate image from sketch"""
    try:
        data = request.json
        sketch_data = data.get('sketch')
        prompt = data.get('prompt', 'detailed artwork, high quality')
        negative_prompt = data.get('negative_prompt', 'blurry, low quality')
        mode = data.get('mode', 'basic')
        
        if not sketch_data:
            return jsonify({'success': False, 'error': 'No sketch data provided'})
        
        # Check if models are loaded
        if pipe is None:
            return jsonify({'success': False, 'error': 'AI models not loaded. Please wait for initialization.'})
        
        # Preprocess sketch
        control_image = preprocess_sketch(sketch_data)
        if control_image is None:
            return jsonify({'success': False, 'error': 'Failed to process sketch'})
        
        # Adjust prompt based on mode
        if mode == 'detailed':
            prompt = f"highly detailed, intricate, {prompt}"
        elif mode == 'artistic':
            prompt = f"artistic masterpiece, {prompt}, vibrant colors"
        
        # Generate image
        generated_image = generate_image_from_sketch(control_image, prompt, negative_prompt)
        if generated_image is None:
            return jsonify({'success': False, 'error': 'Failed to generate image'})
        
        # Convert to base64
        image_base64 = image_to_base64(generated_image)
        
        return jsonify({
            'success': True,
            'image': image_base64,
            'prompt_used': prompt
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status')
def get_status():
    """Check if models are loaded"""
    return jsonify({
        'models_loaded': pipe is not None,
        'cuda_available': torch.cuda.is_available()
    })

if __name__ == '__main__':
    # Initialize models in a separate thread to avoid blocking startup
    import threading
    
    def load_models():
        initialize_models()
    
    model_thread = threading.Thread(target=load_models)
    model_thread.daemon = True
    model_thread.start()
    
    print("Starting Sketch-to-AI Image Generator...")
    print("🎨 Access the app at: http://localhost:5000")
    print("📝 Models are loading in the background...")
    print("⚡ First generation may take longer as models initialize")
    
    app.run(debug=True, host='0.0.0.0', port=5000)