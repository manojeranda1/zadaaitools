from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import torch
import numpy as np
from PIL import Image
from skimage import io as skio
import torch.nn.functional as F
from models.ormbg import ORMBG
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# Config
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
OUTPUT_SIZE = (2000, 2000)
MODEL_INPUT_SIZE = [1024, 1024]
MODEL_PATH = os.path.join("models", "ormbg.pth")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ORMBG()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        torch.unsqueeze(im_tensor, 0), size=model_input_size, mode="bilinear"
    ).type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode="bilinear"), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

def remove_background_with_ormbg(image_stream):
    # Read image and convert
    orig_image = Image.open(image_stream).convert("RGB")
    orig_np = np.array(orig_image)
    orig_size = orig_np.shape[0:2]

    # Preprocess and infer
    image = preprocess_image(orig_np, MODEL_INPUT_SIZE).to(device)
    with torch.no_grad():
        result = model(image)

    # Post-process result mask
    result_mask = postprocess_image(result[0][0], orig_size)

    # Create RGBA image with transparency
    alpha_mask = Image.fromarray(result_mask).convert("L")
    rgba_image = orig_image.convert("RGBA")
    rgba_image.putalpha(alpha_mask)

    return rgba_image

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        result_img = remove_background_with_ormbg(file.stream)

        # Resize and center result
        result_ratio = min(OUTPUT_SIZE[0]/result_img.width, OUTPUT_SIZE[1]/result_img.height)
        new_size = (int(result_img.width * result_ratio), int(result_img.height * result_ratio))
        resized = result_img.resize(new_size, Image.LANCZOS)

        final_image = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
        position = ((OUTPUT_SIZE[0] - new_size[0]) // 2, (OUTPUT_SIZE[1] - new_size[1]) // 2)
        final_image.paste(resized, position, resized)

        img_bytes = io.BytesIO()
        final_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return send_file(
            img_bytes,
            mimetype='image/png',
            download_name=f"nobg_{secure_filename(file.filename)}"
        )
    except Exception as e:
        return jsonify({"error": f"Background removal failed: {str(e)}"}), 500
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
    return jsonify({"status": "ok", "model": "ormbg"})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
