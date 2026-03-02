"""
Tea Leaf Disease Classifier — Hugging Face Space
Swin Transformer model for classifying 12 types of tea leaf diseases.
Optimized for low-light conditions.
"""

import os

# Force Keras 2 legacy mode (required for the SwinTransformer package)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import sys
import numpy as np
import gradio as gr

# Add the local swintransformer package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from swintransformer import SwinTransformer

# ─── Constants ───────────────────────────────────────────────────────────────

IMAGE_SIZE = (224, 224)

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
    "white spot",
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
    "white spot": "White Spot",
}

NUM_CLASSES = len(CLASSES)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_swin.h5")

# ─── Build Model ─────────────────────────────────────────────────────────────


def build_model():
    """Rebuild the exact same architecture used during training."""
    preprocess = tf.keras.layers.Lambda(
        lambda x: tf.keras.applications.imagenet_utils.preprocess_input(
            tf.cast(x, tf.float32), mode="torch"
        ),
        input_shape=[*IMAGE_SIZE, 3],
    )

    swin = SwinTransformer(
        "swin_tiny_224",
        num_classes=NUM_CLASSES,
        include_top=False,
        pretrained=False,
        use_tpu=False,
    )

    model = tf.keras.Sequential(
        [
            preprocess,
            swin,
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    return model


try:
    print("🔨 Building model...")
    model = build_model()

    print(f"📂 Loading weights from: {MODEL_PATH}")
    model.load_weights(MODEL_PATH)
    print("✅ Model loaded successfully!")

    # Warm up the model with a dummy prediction
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
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
        image: Input image as a numpy array (H, W, 3) with values in [0, 255].

    Returns:
        Dictionary mapping class labels to confidence scores.
    """
    if image is None:
        return {}

    if not MODEL_LOADED:
        return {"Error: Model not loaded": 1.0}

    # Resize to the expected input size
    img = tf.image.resize(image, IMAGE_SIZE)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    predictions = model.predict(img, verbose=0)
    probs = predictions[0]

    # Build result dictionary with human-readable labels
    results = {}
    for cls_name, prob in zip(CLASSES, probs):
        display_name = DISPLAY_LABELS.get(cls_name, cls_name)
        results[display_name] = float(prob)

    return results


# ─── Prediction (HTML output) ────────────────────────────────────────────────


def predict_and_format(image):
    """Classify a tea leaf image and return formatted HTML results."""
    results = predict(image)
    if not results:
        return "<p style='color:#888;'>Please upload an image.</p>"

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

with gr.Blocks(theme=gr.themes.Soft(), title="Tea Leaf Disease Classifier") as demo:
    gr.Markdown("""
    # 🍃 Tea Leaf Disease Classifier
    Upload a tea leaf image to identify diseases using a **Swin Transformer** model.
    
    ⚠️ **Important:** This model is specifically trained on tea leaf images. Uploading human faces, animals, or other random objects will produce inaccurate results as the model tries to map them to leaf diseases.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Tea Leaf Image",
                type="numpy",
            )
            submit_btn = gr.Button("🔍 Classify", variant="primary", size="lg")

        with gr.Column():
            output_html = gr.HTML(label="Prediction Results")

    submit_btn.click(
        fn=predict_and_format,
        inputs=image_input,
        outputs=output_html,
        api_name=False,
    )

    # Also trigger on image upload
    image_input.change(
        fn=predict_and_format,
        inputs=image_input,
        outputs=output_html,
        api_name=False,
    )

    gr.Markdown("""
    ### Model Details
    - **Architecture**: Swin Transformer (Tiny, 224×224)
    - **Training**: 25 epochs with AdamW + Cosine Decay LR
    - **Classes**: Algal Spot, Anthracnose, Bird Eye Spot, Brown Blight, Gray Blight,
      Green Mirid Bug, Healthy, Helopeltis, Red Leaf Spot, Red Rust, Red Spider, White Spot

    > For best results, use images with clear visibility of the leaf surface.
    """)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
