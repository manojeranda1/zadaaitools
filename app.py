from flask import Flask, request, jsonify, send_file, render_template
import io
import os
from PIL import Image, ImageFilter, ImageEnhance
import rembg
import logging
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
OUTPUT_SIZE = (2000, 2000)
BACKGROUND_COLOR = "#f7f7f7"
MAX_CONTENT_LENGTH = 10 * 1024 * 1024

# Initialize rembg session
session = rembg.new_session()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Background Removal Processing
def process_bg_removal(image_data):
    try:
        input_image = Image.open(io.BytesIO(image_data)).convert('RGBA')
        output = rembg.remove(input_image, session=session, alpha_matting=False)
        output = output.convert('RGBA')
        
        output_ratio = min(OUTPUT_SIZE[0]/output.width, OUTPUT_SIZE[1]/output.height)
        new_size = (int(output.width * output_ratio), int(output.height * output_ratio))
        
        resized_output = output.resize(new_size, Image.LANCZOS)
        final_image = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
        position = ((OUTPUT_SIZE[0]-new_size[0])//2, (OUTPUT_SIZE[1]-new_size[1])//2)
        final_image.paste(resized_output, position, resized_output)
        
        output_bytes = io.BytesIO()
        final_image.save(output_bytes, format='PNG')
        output_bytes.seek(0)
        return output_bytes
    except Exception as e:
        logger.error(f"BG Removal error: {str(e)}")
        raise

# Image Resizing Processing
def process_resize(image_data):
    try:
        input_image = Image.open(io.BytesIO(image_data))
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        output_ratio = max(OUTPUT_SIZE[0] / input_image.width, OUTPUT_SIZE[1] / input_image.height)
        new_size = (int(input_image.width * output_ratio), int(input_image.height * output_ratio))
        
        resized_image = input_image.resize(new_size, Image.Resampling.HAMMING)
        left = (resized_image.width - OUTPUT_SIZE[0]) // 2
        cropped_image = resized_image.crop((left, 0, left + OUTPUT_SIZE[0], OUTPUT_SIZE[1]))
        
        output_bytes = io.BytesIO()
        cropped_image.save(output_bytes, format='PNG', optimize=True)
        output_bytes.seek(0)
        return output_bytes
    except Exception as e:
        logger.error(f"Resize error: {str(e)}")
        raise

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        processed_image = process_bg_removal(file.read())
        return send_file(
            processed_image,
            mimetype='image/png',
            download_name=f"nobg_{secure_filename(file.filename)}"
        )
    except Exception as e:
        return jsonify({"error": "Background removal failed"}), 500

@app.route('/api/resize-image', methods=['POST'])
def resize_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        processed_image = process_resize(file.read())
        return send_file(
            processed_image,
            mimetype='image/png',
            download_name=f"resized_{secure_filename(file.filename)}"
        )
    except Exception as e:
        return jsonify({"error": "Image resize failed"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "services": ["background-removal", "image-resizing"]})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)