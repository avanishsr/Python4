from flask import Flask, request, jsonify
import cv2
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load the YOLO model
model = YOLO('best.pt')

# Define the upload folder for images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'JPEG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image):
    # Convert the image to a format YOLO expects (BGR, OpenCV format)
    # image_np = np.array(image)
    # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO on the image
    results = model(image)

    # Assuming the first result is the number plate, adjust if needed
    if len(results) > 0 and len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)  # Get first bounding box
        x1, y1, x2, y2 = box

        # # Crop the detected number plate
        # cropped_image = image_np[y1:y2, x1:x2]

        # # Convert to PIL Image for base64 encoding
        # pil_img = Image.fromarray(cropped_image)
        # buffer = BytesIO()
        # pil_img.save(buffer, format="JPEG")
        # img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return 2

    return None

@app.route('/detect_number_plate', methods=['POST'])
def detect_number_plate():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(image_path)

    cropped_image_base64 = process_image(image_path)

    if cropped_image_base64:
        return jsonify({"cropped_image": cropped_image_base64})
    else:
        return jsonify({"error": "Could not detect number plate"}), 400

if __name__ == "__main__":
    app.run(debug=True)
