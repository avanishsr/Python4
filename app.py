from flask import Flask, request, jsonify
import cv2
import base64
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model
model = YOLO("best.pt")

def process_video(file_path):
    # Capture video
    cap = cv2.VideoCapture(file_path)

    # Read the first frame
    ret, frame = cap.read()

    if not ret:
        return None

    # Run YOLO on the first frame
    results = model(frame)

    # Assuming the first result is the number plate, adjust if needed
    if len(results) > 0 and len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)  # Get first bounding box
        x1, y1, x2, y2 = box

        # Crop the detected number plate
        cropped_image = frame[y1:y2, x1:x2]

        # Convert to PIL Image for base64 encoding
        pil_img = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return img_str

    return None

@app.route('/detect_number_plate', methods=['POST'])
def detect_number_plate():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Save the file temporarily
    file_path = "temp_video.mp4"
    file.save(file_path)

    # Process the video
    cropped_image_base64 = process_video(file_path)

    if cropped_image_base64:
        return jsonify({"cropped_image": cropped_image_base64})
    else:
        return jsonify({"error": "Could not detect number plate"}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
