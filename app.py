from flask import Flask, request, jsonify
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")

def process_image(image):
    # Convert the image to a format YOLO expects (BGR, OpenCV format)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLO on the image
    results = model(image_bgr)

    # Assuming the first result is the number plate, adjust if needed
    if len(results) > 0 and len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)  # Get first bounding box
        x1, y1, x2, y2 = box

        # Crop the detected number plate
        cropped_image = image_np[y1:y2, x1:x2]

        # Return bounding box and cropped image
        return {
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "cropped_image": cropped_image
        }

    return None

@app.route('/detect_number_plate', methods=['POST'])
def detect_number_plate():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Open the uploaded image file
    image = Image.open(file)

    # Process the image to detect the number plate
    processed_result = process_image(image)

    if processed_result:
        return jsonify({
            "bounding_box": processed_result["bounding_box"],
            "message": "Number plate detected successfully"
        })
    else:
        return jsonify({"error": "Could not detect number plate"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
