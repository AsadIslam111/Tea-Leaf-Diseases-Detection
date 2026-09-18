"""
Tea Leaf Disease Classifier — Hugging Face Space
Custom PyTorch YOLO11m-Backbone Classifier for 12 types of tea leaf diseases.
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
import google.generativeai as genai

# ─── Constants ───────────────────────────────────────────────────────────────

CLASSES = [
    "algal_spot",
    "anthracnose",
    "bird_eye_spot",
    "brown_blight",
    "gray_blight",
    "green_mirid_bug",
    "healthy",
    "helopeltis",
    "red_leaf_spot",
    "red_rust",
    "red_spider",
    "white_spot",
]

DISPLAY_LABELS = {
    "algal_spot": "Algal Spot",
    "anthracnose": "Anthracnose",
    "bird_eye_spot": "Bird Eye Spot",
    "brown_blight": "Brown Blight",
    "gray_blight": "Gray Blight",
    "green_mirid_bug": "Green Mirid Bug",
    "healthy": "Healthy ✅",
    "helopeltis": "Helopeltis",
    "red_leaf_spot": "Red Leaf Spot",
    "red_rust": "Red Rust",
    "red_spider": "Red Spider",
    "white_spot": "White Spot",
}

NUM_CLASSES = len(CLASSES)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
DEVICE = torch.device("cpu") # Hugging Face basic CPU instance

# ─── Model Architecture ──────────────────────────────────────────────────────

class YOLO11BackboneClassifier(nn.Module):
    def __init__(self, detector_model, num_classes, dropout=0.40, image_size=256):
        super().__init__()
        # Extract backbone feature extraction layers from YOLO11 detector
        self.backbone = nn.ModuleList(list(detector_model.model[:11]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            x = dummy
            for layer in self.backbone:
                x = layer(x)
            feature_channels = x.shape[1]
            
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_channels, num_classes)
        )

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# ─── Preprocessing ──────────────────────────────────────────────────────────

evaluation_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ─── Load Model ─────────────────────────────────────────────────────────────

try:
    print("📂 Downloading standard yolo11m backbone...")
    yolo_detector = YOLO("yolo11m.pt")
    detector_raw = yolo_detector.model.float().eval()
    
    print(f"📂 Building Custom YOLO11m Classifier Architecture...")
    model = YOLO11BackboneClassifier(
        detector_model=detector_raw,
        num_classes=NUM_CLASSES,
        dropout=0.40,
        image_size=256
    ).to(DEVICE)
    
    print(f"📂 Loading Custom trained weights from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Cleanup memory
    del detector_raw, yolo_detector
    
    print("✅ Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None
    MODEL_LOADED = False

# ─── Gemini Setup ───────────────────────────────────────────────────────────

gemini_api_key = os.environ.get("GEMINI_API_KEY")
print(f"🔑 GEMINI_API_KEY present: {bool(gemini_api_key)}, length: {len(gemini_api_key) if gemini_api_key else 0}")
if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
        GEMINI_AVAILABLE = True
        print("✅ Gemini API configured successfully!")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        GEMINI_AVAILABLE = False
else:
    print("⚠️ GEMINI_API_KEY not found in environment variables. OOD filtering disabled.")
    GEMINI_AVAILABLE = False

# ─── Prediction Function ────────────────────────────────────────────────────

def predict(image):
    if image is None:
        return {}
    if not MODEL_LOADED:
        return {"Error: Model not loaded": 1.0}

    # Gradio passes a numpy array; convert to PIL
    try:
        pil_img = Image.fromarray(image)
        
        # --- Gemini OOD Filter ---
        if GEMINI_AVAILABLE:
            try:
                print("🔍 Running Gemini OOD check...")
                response = gemini_model.generate_content(
                    ["Look at this image carefully. Is this image of a plant leaf or part of a plant? If it shows a human, animal, vehicle, electronics, food, or any non-plant object, answer NO. Only answer YES if it clearly shows a leaf or plant. Answer with a single word: YES or NO.", pil_img]
                )
                answer = response.text.strip().upper()
                print(f"🔍 Gemini response: '{answer}'")
                if "NO" in answer:
                    return {"OOD_REJECTED": True}
            except Exception as e:
                print(f"Gemini API check failed: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to YOLO if Gemini fails
        else:
            print("⚠️ Gemini not available, skipping OOD check")
        
        input_tensor = evaluation_transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        result_dict = {}
        # We assume the training class order matches CLASSES alphabetically
        # based on ImageFolder loading
        for idx, prob in enumerate(probs):
            cls_name = CLASSES[idx]
            display_name = DISPLAY_LABELS.get(cls_name, cls_name)
            result_dict[display_name] = float(prob)
        return result_dict
    except Exception as e:
        return {f"Error processing image: {e}": 1.0}

# ─── Prediction (HTML output) ────────────────────────────────────────────────

def predict_and_format(image):
    results = predict(image)
    if not results:
        return "<div style='color:#888; text-align:center; padding: 20px;'>Please upload an image.</div>"
    
    # Check for OOD rejection from Gemini
    if "OOD_REJECTED" in results:
        return """
        <div style='background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); border-radius: 12px; padding: 24px; margin: 10px 0; text-align: center;'>
            <div style='font-size: 48px; margin-bottom: 12px;'>🚫</div>
            <h3 style='color: white; margin: 0 0 8px 0; font-size: 1.3rem;'>Not a Leaf Image</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.95rem;'>
                This image does not appear to be a tea leaf.<br>
                Please upload a clear photo of a tea leaf for accurate disease diagnosis.
            </p>
        </div>
        """
    
    if "Error" in list(results.keys())[0]:
        error_msg = list(results.keys())[0]
        return f"<div style='color:#888; text-align:center; padding: 20px;'>{error_msg}</div>"

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
    top_label, top_conf = sorted_results[0]
    
    THRESHOLD = 0.45
    is_low_confidence = top_conf < THRESHOLD

    html = f"<div style='font-family:sans-serif; padding:8px;'>"
    
    if is_low_confidence:
        html += f"<div style='background:#fff3cd; border-left:4px solid #ffc107; padding:12px; border-radius:4px; margin-bottom:16px;'>"
        html += f"<p style='margin:0; color:#856404; font-size:14px;'>⚠️ <b>Low Confidence:</b> The model is unsure if this is a tea leaf. "
        html += f"Please ensure the image is clear and specifically of a tea leaf surface.</p></div>"
        title_color = "#856404"
    else:
        title_color = "#2d7d46"

    html += f"<h3 style='margin:0 0 16px 0; color:{title_color};'>🍃 {top_label} ({top_conf*100:.1f}%)</h3>"

    for label, conf in sorted_results:
        pct = conf * 100
        bar_color = "#ffc107" if is_low_confidence else ("#2d7d46" if conf == top_conf else "#4a9960")
        
        html += f"""
        <div style='margin-bottom:8px;'>
            <div style='display:flex; justify-content:space-between; margin-bottom:2px;'>
                <span style='font-size:14px; font-weight:500;'>{label}</span>
                <span style='font-size:14px; color:#666;'>{pct:.1f}%</span>
            </div>
            <div style='background:#e8e8e8; border-radius:4px; height:20px; overflow:hidden;'>
                <div style='background:{bar_color}; width:{pct}%; height:100%; border-radius:4px; transition: width 0.3s;'></div>
            </div>
        </div>"""

    html += "</div>"
    return html

# ─── Gradio Blocks App ──────────────────────────────────────────────────────

# JavaScript to permanently force light mode by removing the 'dark' class
force_light_mode_js = """
function() {
    document.body.classList.remove('dark');
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'class' && document.body.classList.contains('dark')) {
                document.body.classList.remove('dark');
            }
        });
    });
    observer.observe(document.body, { attributes: true });
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"), js=force_light_mode_js, title="Tea Leaf Disease Classifier") as demo:
    gr.HTML("""
    <div style='text-align: center; max-width: 800px; margin: 0 auto; padding-top: 10px; padding-bottom: 20px;'>
        <h1 style='color: #2d7d46; font-size: 2.8rem; margin-bottom: 0.5rem;'>🍃 Tea Leaf Disease Classifier</h1>
        <p style='font-size: 1.1rem; color: #555;'>
            An advanced computer vision diagnostic tool powered by a custom <b>YOLO11m Backbone</b> to instantly detect and classify 12 distinct conditions in tea leaves.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Leaf Image")
            gr.Markdown("For best results, upload a clear, focused photo of a single tea leaf.")
            image_input = gr.Image(
                label="",
                type="numpy",
            )
            submit_btn = gr.Button("🔍 Analyze Leaf", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")
            output_html = gr.HTML(value="<div style='color: #999; text-align: center; padding: 40px; border: 2px dashed #eee; border-radius: 8px;'>Upload an image to see the diagnostic analysis.</div>")

    submit_btn.click(
        fn=predict_and_format,
        inputs=image_input,
        outputs=output_html,
        api_name=False,
    )
    image_input.change(
        fn=predict_and_format,
        inputs=image_input,
        outputs=output_html,
        api_name=False,
    )

    with gr.Row():
        with gr.Column():
            with gr.Accordion("📚 About the Model & Dataset", open=False):
                gr.Markdown("""
                ### Model Architecture
                This application runs on a custom classifier built on top of the state-of-the-art **YOLO11m** backbone architecture. It was trained on a robust dataset of over **22,000 augmented images**, explicitly designed to handle real-world agricultural challenges like low-light conditions, heavy shadows, and sensor noise.
                
                **Key Metrics:**
                - **Test Accuracy:** ~96.1%
                - **Classes Detected:** 12
                
                ### The 12 Classifications
                * **Diseases:** Algal Spot, Anthracnose, Bird Eye Spot, Brown Blight, Gray Blight, Helopeltis, Red Leaf Spot, Red Rust, White Spot
                * **Pests/Insects:** Green Mirid Bug, Red Spider
                * **Healthy:** Clean, disease-free leaves
                """)
                
        with gr.Column():
            with gr.Accordion("⚠️ Important Usage Notes", open=False):
                gr.Markdown("""
                - **Out-of-Distribution (OOD) Data:** This model is strictly trained on tea leaves. If you upload a picture of a human face, an animal, or a random object, the model will still mathematically force it into a leaf disease category. A low-confidence warning banner will appear if the model is unsure.
                - **Lighting & Focus:** While the model was trained with advanced low-light augmentation, extremely dark, completely blurry, or highly obscured images will naturally reduce diagnostic accuracy. 
                """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
