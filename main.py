import gradio as gr

def process_image(image):
    # Assume get_predictions is defined elsewhere and returns a label and explanation
    label, explanation = get_predictions(image)
    return label, explanation

# Create the Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Explanation")
    ],
    title="Image Classifier with Explanation",
    description="Upload an image to get a prediction and explanation."
)

# Launch the app
iface.launch()
