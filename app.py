import os
import io
import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf

app = Flask(__name__, static_folder='static')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

# Load the model
model = tf.keras.models.load_model('best_model.keras')
#model = 'test'

# Configure upload folder
UPLOAD_FOLDER = '/tmp/uploads'  # Changed to /tmp for Vercel
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_landmarks_and_draw(image):
    """Extract pose landmarks and draw them on the image"""
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Process image
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, None
    
    # Draw landmarks on the image
    annotated_image = image_cv.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    
    # Extract landmarks
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    
    return np.array(landmarks), annotated_image

def encode_image(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-clip')
def predict_clip():
    return render_template('predict_clip.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        
        # Extract landmarks and get annotated image
        landmarks, annotated_image = extract_landmarks_and_draw(image)
        
        if landmarks is None:
            return jsonify({'error': 'No pose detected in the image'}), 400
        
        # Reshape landmarks for prediction
        landmarks = landmarks.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(landmarks, verbose=0)
        
        # Get result and confidence
        result = 'Fall Detected' if prediction[0][0] > 0.5 else 'No Fall Detected'
        confidence = float(prediction[0][0])
        
        # Convert annotated image to base64
        annotated_image_b64 = encode_image(annotated_image)
        
        return jsonify({
            'prediction': result,
            'confidence': confidence,
            'annotated_image': annotated_image_b64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080))) 