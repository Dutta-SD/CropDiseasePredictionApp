import gradio as gr

from service.predict import generate_prediction_from_image


def process_image(image):
    # Assume get_predictions is defined elsewhere and returns a label and explanation
    label, explanation = generate_prediction_from_image(image)
    return label, explanation


# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(
        image_mode="RGB",
        sources="upload",
        label="Upload Plant Disease Image",
        show_download_button=True,
        type="pil",
    ),
    outputs=[
        gr.Textbox(label="Prediction", placeholder="Disease Prediction"),
        gr.Markdown(label="Remedy"),
    ],
    title="Classify Plant Diseases and Get Remedies",
)

if __name__ == "__main__":
    iface.launch()
