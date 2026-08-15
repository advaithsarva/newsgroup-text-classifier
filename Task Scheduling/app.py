from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Upload folder sits next to this file, so the app runs on any machine.
app.config['UPLOAD_FOLDER'] = os.environ.get(
    'UPLOAD_FOLDER', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Sources')
)
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB file size limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'txt', 'docx','py','html','js','css','sql','ipynb'}

# Check for allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index.html')  # Assumes index.html is in the 'templates' folder

# File upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)
            return jsonify({'message': 'File successfully uploaded'}), 200
        except Exception as e:
            return jsonify({'error': f'File upload failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # debug=True exposes the Werkzeug console (arbitrary code execution) — opt in only.
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
