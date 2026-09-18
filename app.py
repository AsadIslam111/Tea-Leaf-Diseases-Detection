"""
Tea Leaf Disease Classifier — Hugging Face Space
YOLO11 model for classifying 12 types of tea leaf diseases.
"""

import os
import numpy as np
import gradio as gr
from ultralytics import YOLO

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

# Human-readable labels for display
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

# ─── Load Model ─────────────────────────────────────────────────────────────

try:
    print(f"📂 Loading YOLO weights from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")

    # Warm up the model with a dummy prediction
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    model(dummy, verbose=False)
    print("✅ Model warmup complete!")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None
    MODEL_LOADED = False


# ─── Prediction Function ────────────────────────────────────────────────────


def predict(image):
    """
    Classify a tea leaf image.

    Args:
        image: Input image as a numpy array.

    Returns:
        Dictionary mapping class labels to confidence scores.
    """
    if image is None:
        return {}

    if not MODEL_LOADED:
        return {"Error: Model not loaded": 1.0}

    # Run inference
    results = model(image, verbose=False)
    
    if len(results) > 0 and results[0].probs is not None:
        probs = results[0].probs.data.cpu().numpy()
        
        # Build result dictionary with human-readable labels
        # YOLO usually sorts class names alphabetically or by training ID. 
        # `results[0].names` holds the dictionary mapping id -> name
        result_dict = {}
        for idx, prob in enumerate(probs):
            cls_name = results[0].names[idx]
            display_name = DISPLAY_LABELS.get(cls_name, cls_name)
            result_dict[display_name] = float(prob)
        return result_dict
    else:
        return {"Error: Could not predict probabilities": 1.0}


# ─── Prediction (HTML output) ────────────────────────────────────────────────


def predict_and_format(image):
    """Classify a tea leaf image and return formatted HTML results."""
    results = predict(image)
    if not results or "Error" in list(results.keys())[0]:
        error_msg = list(results.keys())[0] if results else "Please upload an image."
        return f"<p style='color:#888;'>{error_msg}</p>"

    # Sort by confidence (descending) and take top 5
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
    top_label, top_conf = sorted_results[0]
    
    # --- Confidence Threshold Check ---
    # If the top prediction is too low, it's likely not a tea leaf image (OOD).
    THRESHOLD = 0.45
    is_low_confidence = top_conf < THRESHOLD

    # Build HTML output with bar chart
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
        # Dim colors if low confidence
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

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"), title="Tea Leaf Disease Classifier") as demo:
    # Header Section
    gr.HTML("""
    <div style='text-align: center; max-width: 800px; margin: 0 auto; padding-top: 10px; padding-bottom: 20px;'>
        <h1 style='color: #2d7d46; font-size: 2.8rem; margin-bottom: 0.5rem;'>🍃 Tea Leaf Disease Classifier</h1>
        <p style='font-size: 1.1rem; color: #555;'>
            An advanced computer vision diagnostic tool powered by <b>YOLO11m</b> to instantly detect and classify 12 distinct conditions in tea leaves.
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

    # Connect the inputs and outputs
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

    # Informational Sections
    with gr.Row():
        with gr.Column():
            with gr.Accordion("📚 About the Model & Dataset", open=False):
                gr.Markdown("""
                ### Model Architecture
                This application runs on the state-of-the-art **YOLO11m** architecture. It was trained on a robust dataset of over **22,000 augmented images**, explicitly designed to handle real-world agricultural challenges like low-light conditions, heavy shadows, and sensor noise.
                
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
