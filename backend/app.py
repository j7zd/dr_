from flask import Flask, request, render_template
import os
from scanner import scan, transform_perspective
import cv2
import numpy as np
import base64
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import BigInteger
import random
import requests
from processing import extract_picture_bg_2024, read_mrz_bg_2024
from face_comp import compare_images

# MySQL configuration
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_PORT = 3306
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASSWORD = None
if MYSQL_USER == 'root':
    MYSQL_PASSWORD = os.environ.get('MYSQL_ROOT_PASSWORD', None)
else:
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', None)
if MYSQL_PASSWORD is None:
    raise ValueError('Failed to get MySQL password')

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Flask SQLAlchemy configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database models
class Corners(db.Model):
    __tablename__ = 'corners'

    id = db.Column(db.Integer, primary_key=True)
    x1 = db.Column(db.Float, nullable=False)
    y1 = db.Column(db.Float, nullable=False)
    x2 = db.Column(db.Float, nullable=False)
    y2 = db.Column(db.Float, nullable=False)
    x3 = db.Column(db.Float, nullable=False)
    y3 = db.Column(db.Float, nullable=False)
    x4 = db.Column(db.Float, nullable=False)
    y4 = db.Column(db.Float, nullable=False)

class Session(db.Model):
    __tablename__ = 'sessions'

    id = db.Column(BigInteger, primary_key=True, autoincrement=False)
    front_corners_id = db.Column(db.Integer, db.ForeignKey('corners.id'), nullable=True)
    back_corners_id = db.Column(db.Integer, db.ForeignKey('corners.id'), nullable=True)
    consistent_count = db.Column(db.Integer, nullable=False)
    previous_size = db.Column(db.PickleType, nullable=True)
    status = db.Column(db.Enum('IN_PROGRESS', 'WAITING', 'DENIED', 'ACCEPTED', name='status_enum'), nullable=False)
    callback_url = db.Column(db.String(255), nullable=False)
    requested_information = db.Column(db.String(255), nullable=True)

    front_corners = db.relationship('Corners', foreign_keys=[front_corners_id], backref=db.backref('front_session', uselist=False), single_parent=True, cascade='all, delete-orphan')
    back_corners = db.relationship('Corners', foreign_keys=[back_corners_id], backref=db.backref('back_session', uselist=False), single_parent=True, cascade='all, delete-orphan')

with app.app_context():
    db.create_all()

# Constants for the scanning process
SIZE_THRESHOLD = 0.1
CONSISTENT_FRAMES = 10

@app.route('/api/verification/start', methods=['POST'])
def verification_start():
    # The callback URL is used to send the result of the verification process
    callback_url = request.json.get('callback_url')
    if not callback_url:
        return "Callback URL is required", 400
    requested_information = request.json.get('requested_information')

    session_id = None
    while True:
        session_id = random.randint(1, 9_223_372_036_854_775_807) # the ids are random to prevent people from guessing someone else's session id
        if  db.session.query(Session).filter_by(id=session_id).first() is None:
            break
    new_session = Session(id=session_id, callback_url=callback_url, consistent_count=0, previous_size=None, status='IN_PROGRESS', requested_information=requested_information)
    db.session.add(new_session)
    db.session.commit()

    return {"session_id": session_id}, 201

@app.route('/api/verification/cancel/<session_id>', methods=['DELETE'])
def verification_cancel(session_id):
    session = db.session.query(Session).filter_by(id=session_id).first()
    if session is None:
        return "Session not found", 404
    db.session.delete(session)
    db.session.commit()

    return "Session deleted", 200

@app.route('/api/scan/add/<session_id>', methods=['POST'])
def add_scan(session_id):
    session = db.session.query(Session).filter_by(id=session_id).first()
    if session is None:
        return "Session not found", 404
    if session.consistent_count >= CONSISTENT_FRAMES:
        return "Scan already finished", 400
    image = request.json['image']
    image = base64.b64decode(image)
    image = np.frombuffer(image, np.uint8)
    finished = False

    corners, transformed_image = scan(image)
    # bellow is the logic that checks for a number of consistent frames. When the required number is reached the scan is successful
    if transformed_image is not None:
        current_size = transformed_image.shape[:2]
        previous_size = session.previous_size
        if previous_size is not None:
            size_diff = np.abs(np.array(current_size) - np.array(previous_size)) / np.array(previous_size)
            if np.all(size_diff < SIZE_THRESHOLD):
                session.consistent_count += 1
                if session.consistent_count >= CONSISTENT_FRAMES:
                    finished = True
                    corners_obj = Corners(x1=corners[0, 0], y1=corners[0, 1], x2=corners[1, 0], y2=corners[1, 1], x3=corners[2, 0], y3=corners[2, 1], x4=corners[3, 0], y4=corners[3, 1])
                    db.session.add(corners_obj)
                    if session.front_corners is None:
                        session.front_corners = corners_obj
                        session.consistent_count = 0
                    else:
                        session.back_corners = corners_obj
            else:
                session.consistent_count = 0
        session.previous_size = current_size

        db.session.commit()
    
        return {
            'test': session.consistent_count,
            'finished': finished,
            'corners': corners.tolist(),
            'transformed_image': base64.b64encode(cv2.imencode('.jpg', transformed_image)[1].tobytes()).decode('utf-8')
        }
    
    return {
        'finished': finished
    }
     
@app.route('/api/scan/restart/<session_id>', methods=['POST'])
def restart_scan(session_id): # the user can restart the scan if they accidentally scanned the wrong object
    session = db.session.query(Session).filter_by(id=session_id).first()
    if session is None:
        return "Session not found", 404
    if session.back_corners is None:
        session.front_corners = None
    else:
        session.back_corners = None
    session.consistent_count = 0
    session.previous_size = None
    db.session.commit()

    return "Scan restarted", 200

@app.route('/api/scan/confirm/<session_id>', methods=['POST'])
def confirm_scan(session_id):
    session = db.session.query(Session).filter_by(id=session_id).first()
    if session is None:
        return "Session not found", 404
    if session.consistent_count < CONSISTENT_FRAMES:
        return "Scan not finished", 400
    
    # Extract the images from the request and decode them
    front_image = request.files['front_image']
    back_image = request.files['back_image']
    face_image = request.files['face_image']
    front_image_data = np.frombuffer(front_image.read(), np.uint8)
    back_image_data = np.frombuffer(back_image.read(), np.uint8)
    face_image_data = np.frombuffer(face_image.read(), np.uint8)
    front_image = cv2.imdecode(front_image_data, cv2.IMREAD_COLOR)
    back_image = cv2.imdecode(back_image_data, cv2.IMREAD_COLOR)
    face_image = cv2.imdecode(face_image_data, cv2.IMREAD_COLOR)

    c = session.front_corners
    corners = np.array([[c.x1, c.y1], [c.x2, c.y2], [c.x3, c.y3], [c.x4, c.y4]])
    corners = corners * [front_image.shape[1] / 640, front_image.shape[0] / 400]
    front_transformed_image = transform_perspective(front_image, corners)

    c = session.back_corners
    corners = np.array([[c.x1, c.y1], [c.x2, c.y2], [c.x3, c.y3], [c.x4, c.y4]])
    corners = corners * [back_image.shape[1] / 640, back_image.shape[0] / 400]
    back_transformed_image = transform_perspective(back_image, corners)

    picture_card = extract_picture_bg_2024(front_transformed_image)
    mrz_data = read_mrz_bg_2024(back_transformed_image)
    l2_distance = compare_images(face_image, picture_card)

    callback_url = session.callback_url
    if l2_distance > 0.4:
        response = requests.post(callback_url, json={'status': 'DENIED'})
    else:
        response = requests.post(callback_url, json={'status': 'ACCEPTED'})

    return {
        'front_transformed_image': base64.b64encode(cv2.imencode('.jpg', front_transformed_image)[1].tobytes()).decode('utf-8'),
        'back_transformed_image': base64.b64encode(cv2.imencode('.jpg', back_transformed_image)[1].tobytes()).decode('utf-8')
    }

@app.route('/api/verification/check_status/<session_id>', methods=['GET'])
def check_status(session_id):
    session = db.session.query(Session).filter_by(id=session_id).first()
    if session is None:
        return "Session not found", 404

    return {
        'status': session.status,
        'requested_information': session.requested_information
    }

@app.route('/api/verification/request_review/<session_id>', methods=['POST'])
def request_review():
    pass

@app.route('/api/admin/review', methods=['GET'])
def review_list():
    pass

@app.route('/api/admin/review/<session_id>', methods=['GET', 'POST'])
def review(session_id):
    pass

@app.route('/verify/<session_id>/start', methods=['GET'])
def start_page(session_id):
    return render_template('start.html', session_id=session_id)

@app.route('/verify/<session_id>/scan', methods=['GET'])
def scan_page(session_id):
    return render_template('scan.html', session_id=session_id)

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)