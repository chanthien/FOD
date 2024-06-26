from flask import Flask, request, render_template, redirect, url_for, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
import cv2
import os
import yt_dlp
from utils import load_model, detect_objects, transform_to_gps, perspective_matrix

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')



# Use absolute path for database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database/data.db')
db = SQLAlchemy(app)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    object_id = db.Column(db.String(50))
    object_name = db.Column(db.String(50))
    gps_lat = db.Column(db.Float)
    gps_lon = db.Column(db.Float)
    confidence = db.Column(db.Float)

# Ensure database tables are created within the app context
with app.app_context():
    db.create_all()

# Load YOLO model
model_path = 'models/yolov8n.pt'
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('begin.html')

@app.route('/process', methods=['POST'])
def process_input():
    input_type = request.form.get('input_type')
    input_data = request.form.get('input_data')
    
    if input_type == 'webcam':
        source = "0"
    elif input_type == 'ip_camera':
        source = input_data
    elif input_type == 'video_url':
        source = download_video(input_data)
    elif input_type == 'video_file':
        source = os.path.join('uploads', input_data)
    else:
        return "Invalid input type", 400

    return redirect(url_for('main', source=source))

@app.route('/main')
def main():
    source = request.args.get('source')
    if source is None:
        return "No video source provided", 400
    return render_template('main.html', source=source)

# @app.route('/detect', methods=['POST'])
# def detect():
#     source = request.form.get('source')
#     if source is None:
#         return "No video source provided", 400
#     try:
#         frame = next(capture_frame(source))
#     except ValueError as e:
#         return str(e), 400
#     detections = detect_objects(model, frame)
#     gps_data = transform_to_gps(detections, perspective_matrix)
#     save_detections(gps_data)

#     # Emit the results to the client using SocketIO
#     socketio.emit('update', gps_data)

#     return jsonify(gps_data)

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    source = data.get('source')
    if source is None:
        return "No video source provided", 400
    try:
        frame = next(capture_frame(source))
    except ValueError as e:
        return str(e), 400
    detections = detect_objects(model, frame)
    gps_data = transform_to_gps(detections, perspective_matrix)
    save_detections(gps_data)

    # Emit the results to the client using SocketIO
    socketio.emit('update', gps_data)

    return jsonify(gps_data)


@app.route('/video_feed')
# def video_feed():
#     source = request.args.get('source')
#     if source is None:
#         return "No video source provided", 400
#     return Response(capture_frame(source), mimetype='multipart/x-mixed-replace; boundary=frame')

def video_feed():
    source = request.args.get('source')
    if source  is None:
        return "No video source provided", 400
    return Response(VIDEO.show(source), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_frame(source):
    print(f"Capture frame source: {source}")  # Debug statement
    if source == "0":  # Webcam
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video source: {source}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to capture frame")
        frame = cv2.imencode(".jpg", frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def download_video(url):
    ydl_opts = {
        'format': 'best',
        "quiet": True,
        "no_warnings": True,
        "forceurl": True,
    }
    ydl = yt_dlp.YoutubeDL(ydl_opts)
    # Extract the video URL
    info = ydl.extract_info(url, download=False)
    url = info["url"]
    return url

def save_detections(detections):
    for det in detections:
        new_detection = Detection(
            object_id=det['id'],
            object_name=det['name'],
            gps_lat=det['gps']['lat'],
            gps_lon=det['gps']['lon'],
            confidence=det['confidence']
        )
        db.session.add(new_detection)
    db.session.commit()

if __name__ == '__main__':
    socketio.run(app, debug=True)
