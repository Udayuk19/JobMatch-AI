from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from utils.ranker import rank_resumes

app = Flask(__name__)

# Upload folder and allowed extensions
UPLOAD_FOLDER = 'resumes'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the resume upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Utility function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return "No file part in the request", 400

    file = request.files['resume']

    if file.filename == '':
        return "No file selected", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('results', filename=filename))

    return "Invalid file type. Only PDF files are allowed.", 400

# Results route to display rankings and suggestions
@app.route('/results/<filename>')
def results(filename):
    try:
        ranked_resumes = rank_resumes(filename)
        return render_template('results.html', ranks=ranked_resumes)
    except Exception as e:
        return f"Error processing the resume: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
