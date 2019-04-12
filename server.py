from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os

from src.eval import predict

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
  if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            print('No file part')
            return redirect(url_for('index'))
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('viewPhoto',
                                filename=filename))


@app.route('/view', methods=['GET'])
def viewPhoto():
  return 'successful file upload'

app.run(host='0.0.0.0', port=5000, debug=True)