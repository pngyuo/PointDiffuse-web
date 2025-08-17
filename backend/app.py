import os
import time
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from model_utils import load_model, process_point_cloud
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['ALLOWED_EXTENSIONS'] = {'npy', 'npz'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# 加载模型
model = load_model()


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Backend is running'}), 200


@app.route('/process', methods=['POST'])
def process_file():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        try:
            # 处理点云
            results = process_point_cloud(model, file_path)
            processing_time = time.time() - start_time

            # 保存可视化结果
            viz_filename = f"viz_{timestamp}_{filename}.png"
            viz_path = os.path.join(app.config['UPLOAD_FOLDER'], viz_filename)
            results['visualization'].save(viz_path)

            # 清理上传的原文件
            try:
                os.remove(file_path)
            except:
                pass

            return jsonify({
                'status': 'success',
                'processing_time': round(processing_time, 2),
                'class_distribution': results['class_distribution'],
                'visualization': f"/download/{viz_filename}",
                'points_count': results['points_count']
            })

        except Exception as e:
            # 清理上传的文件
            try:
                os.remove(file_path)
            except:
                pass

            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/download/<filename>')
def download_file(filename):
    # 确保文件名安全
    safe_filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    print("Starting PointDiffuse backend server...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"Model loaded: {type(model).__name__}")
    app.run(host='0.0.0.0', port=5000, debug=False)