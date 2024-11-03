import gradio as gr

from service.predict import workflow


def process_image(image):
    disease_name, remedy = workflow(image)
    return disease_name, remedy


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
