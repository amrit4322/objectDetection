from flask import Flask, render_template, jsonify, request, send_file, redirect
from ultralytics import YOLO
from flask_socketio import SocketIO
import cv2
import base64
import threading
import os
import urllib.request
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)
model = YOLO('yolov8s.pt') 
stop_thread = False
thread_lock = threading.Lock()
urllib.request.urlretrieve("https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true", "coco.names")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Render the main HTML template
@app.route("/")
def index():
    return render_template("index.html")

# Handle file upload and initiate object detection
@app.route('/objectDetectionFile', methods=['POST'])
def object_detection_file():
    file = request.files['file']
    
    if file:
        # Perform object detection logic on the uploaded file
        file.save("uploads/" + file.filename)
        global stop_thread
        stop_thread = False
        # Start a background thread for object detection
        socketio.start_background_task(target=object_detection_thread, source=f"uploads/{file.filename}", save=True)
        result = f"Object detection from file: {file.filename}"
        return redirect("/")
    else:
        return redirect("/")

# Capture frames from the client's webcam via WebSocket
@socketio.on('capture_frame')
def capture_frame(data):
    global stop_thread
    stop_thread=False
    # Decode base64 image data received from the client
    image_data = data.replace('data:image/jpeg;base64,', '')
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Process the captured frame
    object_detection_thread_web(image_np)
    return "Ok", 200

# Stop object detection on the server
@app.route('/stopObjectDetection')
def stop_object_detection():
    global stop_thread
    with thread_lock:
        stop_thread = True
    return redirect("/")

# Function for object detection on frames received via WebSocket
def object_detection_thread_web(frame):
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    # Perform object detection using YOLO on the frame
    results = model.predict(frame)
    for result in results:
        xyxys = result.boxes.xyxy
        class_id = result.boxes.cls
        confidences = result.boxes.conf
        for i, xyxy in enumerate(xyxys):
            if confidences[i] > 0.5:
                label = str(classes[int(class_id[i])])
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3) 
                cv2.putText(frame, f"{label}: {confidences[i]:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Encode the annotated frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer)
    if stop_thread == False:
        # Send the base64-encoded frame to the client via WebSocket
        socketio.emit('update_frame', {'image': encoded_frame.decode('utf-8')})
    else:
        # Notify the client that object detection is completed
        socketio.emit("completed_web", {'value': 'done'})

# Function for object detection on frames from webcam or video file
def object_detection_thread(source=0, save=False):
    global stop_thread
    cap = cv2.VideoCapture(source)
    if save:
        # Setup video writer for saving the processed frames
        output_path = os.path.join("predictions", "result_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
       
    while stop_thread == False:
        ret, frame = cap.read()
        
        if not ret:
            break
        if save == False:
            frame = cv2.flip(frame, 1)
        # Perform YOLOv8 inference on the frame
        results = model.predict(frame)
        
        for result in results:
            xyxys = result.boxes.xyxy
            class_id = result.boxes.cls
            confidences = result.boxes.conf
            for i, xyxy in enumerate(xyxys):
                if confidences[i] > 0.5:
                    label = str(classes[int(class_id[i])])
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3) 
                    cv2.putText(frame, f"{label}: {confidences[i]:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save:
            output_video.write(frame)
        # Encode the annotated frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer)

        # Send the base64-encoded frame to the client via WebSocket
        socketio.emit('update_frame', {'image': encoded_frame.decode('utf-8')})
        
        # Optionally, perform other real-time processing or visualization here
    
    cap.release()
    if save:
        output_video.release()
    socketio.emit("completed", {'value': 'done'})
    cv2.destroyAllWindows()

# Route for downloading the processed video
@app.route('/download')
def download():
    file_path = "predictions/result_video.mp4"
    mime_type = 'video/mp4'
    filename = os.path.basename(file_path)
    return send_file(file_path, mimetype=mime_type, as_attachment=True, download_name=filename)

if __name__ == '__main__':
    socketio.run(app, debug=False, use_reloader=False, allow_unsafe_werkzeug=True, host='0.0.0.0', port=5000)
