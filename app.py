from flask import Flask, render_template, request, send_file, url_for, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from png2audioR06dev import png2audio
from audio2pngR06dev import audio2png
from song_identifier import identify_song  
from pathlib import Path
from datetime import datetime
import time
import logging  # Import the logging module
import cProfile


app_folder = Path(__file__).resolve().parent
os.chdir(app_folder)
UPLOAD_FOLDER = Path('uploads').resolve()
OUTPUT_FOLDER = Path('output').resolve() 

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp3', 'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


# Set up the logger
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)
logging.basicConfig(level=logging.DEBUG)
import threading


    
def delete_old_files():
    age_threshold = 7*24*60*60  # 7 days
    now = time.time()
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.getatime(file_path) < now - age_threshold:
            os.remove(file_path)

    for filename in os.listdir(app.config['OUTPUT_FOLDER']):
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.getatime(file_path) < now - age_threshold:
            os.remove(file_path)

    # call this function again in 24 hours
    threading.Timer(24*60*60, delete_old_files).start()
    
def initialize():
    delete_old_files()
initialize()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def home():
    pr = cProfile.Profile()
    pr.enable()
    app.logger.info('Info level log')
    app.logger.warning('Warning level log')
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify(error='No file part'), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify(error='No selected file'), 400
        if file and allowed_file(file.filename):
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = secure_filename(f"{timestamp}_{file.filename}")
            filepath = app.config['UPLOAD_FOLDER'] / filename
            file.save(filepath)
            app.logger.info('File uploaded: %s', filename)  # Log the file upload
            extension = filename.rsplit('.', 1)[1].lower()
            
            if extension in ['png', 'jpg', 'jpeg']:
                output_filename = f"{timestamp}_output.mp3"
                output_filepath = app.config['OUTPUT_FOLDER'] / output_filename
                app.logger.info('starting png2audio')
                png2audio(filepath, output_filepath)
                app.logger.info('png2audio complete')
                song_name, artist = identify_song(output_filepath)
                return jsonify({
                    'song_url': url_for('send_file', filename=output_filename, _external=True),
                    'song_name': song_name,
                    'artist': artist,
                    'image_url': url_for('send_file_from_uploads', filename=filename, _external=True)  # Return the URL of the uploaded image
                })
            elif extension in ['mp3', 'wav']:
                output_filename = f"{timestamp}_output.png"
                output_filepath = app.config['OUTPUT_FOLDER'] / output_filename
                app.logger.info('starting audio2png')
                audio2png(filepath, output_filepath)
                app.logger.info('audio2png complete')
                # Identify the song from the uploaded audio file
                # song_name, artist = identify_song(filepath)
                return jsonify({
                    'image_url': url_for('send_file', filename=output_filename, _external=True),
                    #'song_name': song_name,  # Return the identified song name
                    #'artist': artist  # Return the identified artist
                })
    pr.disable()
    pr.print_stats(sort='time')
    return render_template('home.html')


@app.route('/uploads/<filename>')
def send_file_from_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    if file and allowed_file(file.filename):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        if not os.path.exists(filepath):
            app.logger.error(f"File does not exist: {filepath}")
            return
        try:
            with open(filepath, 'rb') as file:
                app.logger.info(f"File can be opened: {filepath}")
        except Exception as e:
            app.logger.error(f"File cannot be opened: {filepath}. Error: {e}")
        

        extension = filename.rsplit('.', 1)[1].lower()
        if extension in ['png', 'jpg', 'jpeg']:
            return jsonify({
                'image_url': url_for('send_file_from_uploads', filename=filename, _external=True),
                'filename': filename
            })
        elif extension in ['mp3', 'wav']:
            try:
                song_name, artist = identify_song(filepath)
            except Exception as e:
                app.logger.error("Error during song identification:", str(e))
            #song_name, artist = identify_song(filepath)
            return jsonify({
                'song_url': url_for('send_file_from_uploads', filename=filename, _external=True),
                'filename': filename,
                'song_name': song_name,  # Return the identified song name
                'artist': artist  # Return the identified artist
            })
            

@app.route('/process', methods=['POST'])
def process_file():
    filename = request.form['filename']
    filepath = app.config['UPLOAD_FOLDER'] / filename
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # Run the appropriate conversion script based on the file extension
    extension = filename.rsplit('.', 1)[1].lower()
    
    if extension in ['png', 'jpg', 'jpeg']:
        output_filename = f"{timestamp}_output.mp3"
        output_filepath = app.config['OUTPUT_FOLDER'] / output_filename

        png2audio(filepath, output_filepath)

        song_name, artist = identify_song(output_filepath)
        return jsonify({
            'song_url': url_for('send_file', filename=output_filename, _external=True),
            'song_name': song_name,
            'artist': artist
        })
    elif extension in ['mp3', 'wav']:
        output_filename = f"{timestamp}_output.png"
        output_filepath = app.config['OUTPUT_FOLDER'] / output_filename

        audio2png(filepath, output_filepath)
        return jsonify({
            'image_url': url_for('send_file', filename=output_filename, _external=True),
        })


@app.route('/send_file/<filename>')
def send_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

    
if __name__ == "__main__":
    app.run(debug=True)
