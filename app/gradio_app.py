import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from app.model import run_pipeline


def predict_ui(image: Image.Image):
    if image is None:
        return None, "Please upload an image."

    result = run_pipeline(image)

    if result["status"] == "no_plate_found":
        return image, "❌ No license plate detected."

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    output_lines = []

    for i, plate in enumerate(result["plates"]):
        box = plate["bounding_box"]
        text = plate["plate_text"]

        # Draw rectangle around detected plate
        draw.rectangle(
            [box["x1"], box["y1"], box["x2"], box["y2"]],
            outline="red", width=3
        )

        # Build output text for display
        output_lines.append(f"Plate {i+1}:")
        output_lines.append(text)
        output_lines.append("")  # empty line between plates

    return image, "\n".join(output_lines)


with gr.Blocks(title="Egyptian Plate Recognition") as demo:
    gr.Markdown("# 🚗 Egyptian Car Plate Recognition")
    gr.Markdown("Upload a car image to detect and extract the license plate text.")

    with gr.Row():
        input_image  = gr.Image(type="pil", label="Upload Car Image")
        output_image = gr.Image(type="pil", label="Detected Plate")

    output_text = gr.Textbox(
        label="Extracted Plate Text (separated characters)",
        lines=4,
        rtl=True          # ← right-to-left for Arabic text display
    )

    submit_btn = gr.Button("🔍 Detect Plate", variant="primary")
    submit_btn.click(
        fn=predict_ui,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

if __name__ == "__main__":
    demo.launch()