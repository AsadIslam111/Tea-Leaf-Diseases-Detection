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

gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if gemini_api_key:
    gemini_api_key = gemini_api_key.strip()

print(f"🔑 GEMINI_API_KEY present: {bool(gemini_api_key)}, length: {len(gemini_api_key) if gemini_api_key else 0}")

GEMINI_AVAILABLE = False
GEMINI_STATUS_LABEL = "Offline (Key Missing)"
CANDIDATE_MODELS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash"]
active_model_name = "gemini-3.6-flash"

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        GEMINI_AVAILABLE = True
        GEMINI_STATUS_LABEL = "Active (Gemini AI)"
        print("✅ Gemini API configured successfully!")
    except Exception as e:
        print(f"❌ Error configuring Gemini API: {e}")
        GEMINI_STATUS_LABEL = f"Config Error: {e}"
        GEMINI_AVAILABLE = False
else:
    print("⚠️ GEMINI_API_KEY not found in environment variables. OOD filtering disabled.")
    GEMINI_STATUS_LABEL = "Offline (GEMINI_API_KEY missing in Space Secrets)"
    GEMINI_AVAILABLE = False

# ─── Prediction Function ────────────────────────────────────────────────────

def predict(image):
    global active_model_name
    if image is None:
        return {}
    if not MODEL_LOADED:
        return {"Error: Model not loaded": 1.0}

    try:
        # Ensure image is in RGB format (handles RGBA PNGs or palette images)
        pil_img = Image.fromarray(image).convert("RGB")
        
        # --- Gemini OOD Filter ---
        if GEMINI_AVAILABLE:
            try:
                print(f"🔍 Running Gemini OOD check (trying {active_model_name})...")
                prompt = (
                    "You are a strict plant leaf detector. Examine this image carefully.\n"
                    "Question: Is the main subject of this image clearly a plant leaf, tea leaf, or plant foliage?\n"
                    "- If the image contains a human, person, face, animal, computer, phone, electronic device, furniture, vehicle, cartoon, meme, or any other non-plant object, you MUST answer NO.\n"
                    "- Only answer YES if the image clearly and primarily shows a plant leaf or foliage.\n"
                    "Answer ONLY with a single word: YES or NO."
                )
                
                # Try active model, fallback to others if model is retired/unavailable
                response = None
                last_err = None
                models_to_try = [active_model_name] + [m for m in CANDIDATE_MODELS if m != active_model_name]
                
                for m_name in models_to_try:
                    try:
                        g_model = genai.GenerativeModel(m_name)
                        response = g_model.generate_content(
                            [prompt, pil_img],
                            generation_config={"temperature": 0.0, "max_output_tokens": 10},
                        )
                        active_model_name = m_name
                        break
                    except Exception as err:
                        last_err = err
                        print(f"⚠️ Model {m_name} failed: {err}")
                
                if response is None:
                    raise last_err
                
                # Check response text safely
                answer = ""
                try:
                    answer = response.text.strip().upper()
                except Exception as text_err:
                    print(f"⚠️ Could not read response.text: {text_err}")
                    # If candidate was blocked by safety filters (e.g. violent/controversial meme), it's not a leaf
                    return {"OOD_REJECTED": True}
                
                print(f"🔍 Gemini response: '{answer}'")
                
                # If Gemini says NO, or does not say YES, reject as non-leaf
                if "NO" in answer or "YES" not in answer:
                    return {"OOD_REJECTED": True}
            except Exception as e:
                print(f"Gemini API check failed: {e}")
                import traceback
                traceback.print_exc()
                return {"GEMINI_API_ERROR": str(e)}
        else:
            print("⚠️ Gemini not available, skipping OOD check")
        
        input_tensor = evaluation_transform(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        result_dict = {}
        for idx, prob in enumerate(probs):
            cls_name = CLASSES[idx]
            display_name = DISPLAY_LABELS.get(cls_name, cls_name)
            result_dict[display_name] = float(prob)
        
        if not GEMINI_AVAILABLE:
            result_dict["_GEMINI_OFFLINE_NOTICE"] = True
            
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
        <div style='background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); border-radius: 12px; padding: 24px; margin: 10px 0; text-align: center; box-shadow: 0 4px 12px rgba(204,0,0,0.15);'>
            <div style='font-size: 48px; margin-bottom: 12px;'>🚫</div>
            <h3 style='color: white; margin: 0 0 8px 0; font-size: 1.3rem; font-weight: 700;'>Not a Leaf Image</h3>
            <p style='color: rgba(255,255,255,0.95); margin: 0; font-size: 0.95rem; line-height: 1.5;'>
                This image was identified as non-plant content (e.g., person, electronic device, meme, or object).<br>
                Please upload a clear, focused photo of a tea leaf for accurate disease diagnosis.
            </p>
        </div>
        """

    if "GEMINI_API_ERROR" in results:
        err = results["GEMINI_API_ERROR"]
        return f"""
        <div style='background:#fff3cd; border-left:4px solid #ffc107; padding:16px; border-radius:8px; margin:10px 0;'>
            <h4 style='margin:0 0 6px 0; color:#856404;'>⚠️ Gemini API Verification Error</h4>
            <p style='margin:0 0 8px 0; color:#856404; font-size:13px;'>Could not verify if this image is a leaf via Gemini API: <code>{err}</code></p>
            <p style='margin:0; color:#666; font-size:12px;'>Please check your <code>GEMINI_API_KEY</code> in Hugging Face Space Settings.</p>
        </div>
        """
    
    # Extract offline flag if present
    gemini_offline = results.pop("_GEMINI_OFFLINE_NOTICE", False)

    if "Error" in list(results.keys())[0]:
        error_msg = list(results.keys())[0]
        return f"<div style='color:#888; text-align:center; padding: 20px;'>{error_msg}</div>"

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
    top_label, top_conf = sorted_results[0]
    
    THRESHOLD = 0.45
    is_low_confidence = top_conf < THRESHOLD

    html = f"<div style='font-family:sans-serif; padding:8px;'>"
    
    if gemini_offline:
        html += """
        <div style='background:#fff8e1; border-left:4px solid #ffa000; padding:10px 14px; border-radius:6px; margin-bottom:14px;'>
            <div style='font-weight:600; color:#b78103; font-size:13px;'>⚠️ Non-Leaf Filter (OOD) is Offline</div>
            <div style='color:#6d4c41; font-size:12px; margin-top:2px;'>
                <code>GEMINI_API_KEY</code> is not configured in Space Secrets. Non-leaf filtering is disabled until key is added.
            </div>
        </div>
        """

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

status_badge = (
    "<span style='background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; padding:5px 14px; border-radius:20px; font-size:0.85rem; font-weight:600; display:inline-block;'>🛡️ Non-Leaf Filter: Active (Gemini AI)</span>"
    if GEMINI_AVAILABLE else
    "<span style='background:#fff3e0; color:#e65100; border:1px solid #ffcc80; padding:5px 14px; border-radius:20px; font-size:0.85rem; font-weight:600; display:inline-block;'>⚠️ Non-Leaf Filter: Offline (GEMINI_API_KEY Missing in Space Secrets)</span>"
)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"), js=force_light_mode_js, title="Tea Leaf Disease Classifier") as demo:
    gr.HTML(f"""
    <div style='text-align: center; max-width: 800px; margin: 0 auto; padding-top: 10px; padding-bottom: 20px;'>
        <h1 style='color: #2d7d46; font-size: 2.8rem; margin-bottom: 0.5rem;'>🍃 Tea Leaf Disease Classifier</h1>
        <p style='font-size: 1.1rem; color: #555; margin-bottom: 12px;'>
            An advanced computer vision diagnostic tool powered by a custom <b>YOLO11m Backbone</b> to instantly detect and classify 12 distinct conditions in tea leaves.
        </p>
        <div>{status_badge}</div>
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
