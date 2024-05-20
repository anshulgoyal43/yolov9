from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageDraw
import io
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="oblGd9IeJj8Hm0rKfO2v"
)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload Image for Annotation</title>
    <h1>Upload Image</h1>
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Read the image
        image = Image.open(file.stream)

        # Inference
        result = CLIENT.infer(image, model_id="signature-detection-xxdzl/3")

        # Draw the predictions on the image
        draw = ImageDraw.Draw(image)
        for prediction in result['predictions']:
            x = prediction['x']
            y = prediction['y']
            width = prediction['width']
            height = prediction['height']
            confidence = prediction['confidence']
            label = prediction['class']

            # Calculate the bounding box coordinates
            left = x - width / 2
            top = y - height / 2
            right = x + width / 2
            bottom = y + height / 2

            # Draw the bounding box
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

            # Draw the label and confidence
            draw.text((left, top - 10), f"{label} ({confidence:.2f})", fill="red")

        # Save the image to a BytesIO object
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG')
        img_io.seek(0)

        # Return the image
        return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)

