from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = r'C:\Projects\Project SB\Task Scheduling\Sources'  # Update this path as needed
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 16MB file size limit

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
    app.run(debug=True)
