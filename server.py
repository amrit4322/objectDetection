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

focal_length = 1000 #will adjust this later

# Placeholder values for average heights (in meters)
average_heights = {
    "person": 1.7,
    "bicycle": 1.0,
    "car": 1.5,
    "motorcycle": 1.2,
    "airplane": 3.0,
    "bus": 3.5,
    "train": 3.0,
    "truck": 3.5,
    "boat": 1.0,
    "traffic light": 0.2,
    "fire hydrant": 1.0,
    "stop sign": 1.5,
    "parking meter": 0.8,
    "bench": 0.5,
    "bird": 0.3,
    "cat": 0.25,
    "dog": 0.3,
    "horse": 1.8,
    "sheep": 0.9,
    "cow": 1.4,
    "elephant": 3.0,
    "bear": 1.6,
    "zebra": 1.5,
    "giraffe": 5.0,
    "backpack": 0.4,
    "umbrella": 0.9,
    "handbag": 0.2,
    "tie": 0.2,
    "suitcase": 0.5,
    "frisbee": 0.2,
    "skis": 1.8,
    "snowboard": 1.6,
    "sports ball": 0.2,
    "kite": 1.0,
    "baseball bat": 0.8,
    "baseball glove": 0.2,
    "skateboard": 0.2,
    "surfboard": 1.8,
    "tennis racket": 0.7,
    "bottle": 0.2,
    "wine glass": 0.2,
    "cup": 0.1,
    "fork": 0.2,
    "knife": 0.2,
    "spoon": 0.2,
    "bowl": 0.2,
    "banana": 0.2,
    "apple": 0.2,
    "sandwich": 0.1,
    "orange": 0.2,
    "broccoli": 0.3,
    "carrot": 0.2,
    "hot dog": 0.1,
    "pizza": 0.2,
    "donut": 0.1,
    "cake": 0.3,
    "chair": 0.5,
    "couch": 0.7,
    "potted plant": 0.4,
    "bed": 0.6,
    "dining table": 0.8,
    "toilet": 0.4,
    "tv": 0.3,
    "laptop": 0.2,
    "mouse": 0.1,
    "remote": 0.1,
    "keyboard": 0.1,
    "cell phone": 0.1,
    "microwave": 0.3,
    "oven": 0.4,
    "toaster": 0.1,
    "sink": 0.2,
    "refrigerator": 0.6,
    "book": 0.3,
    "clock": 0.2,
    "vase": 0.4,
    "scissors": 0.1,
    "teddy bear": 0.5,
    "hair drier": 0.1,
    "toothbrush": 0.1
}
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

# Function to calculate distance between two objects
def calculate_distance(known_height, focal_length, box_height):
    return (known_height * focal_length) / box_height

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
                average_height = average_heights.get(label, 1.0)  # Use 1.0 as a default if not found

                # Calculate distance
                box_height = int(xyxy[3]) - int(xyxy[1])
                distance = calculate_distance(average_height, focal_length, box_height)

                        # Display distance on the frame
                # cv2.putText(frame, f"Distance: {round(distance, 2)} meters", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
                    average_height = average_heights.get(label, 1.0)  # Use 1.0 as a default if not found

                    # Calculate distance
                    box_height = int(xyxy[3]) - int(xyxy[1])
                    distance = calculate_distance(average_height, focal_length, box_height)

                    # Display distance on the frame
                    # cv2.putText(frame, f"Distance: {round(distance, 2)} meters", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3) 
                    cv2.putText(frame, f"{label}: {round(distance, 2)} meters", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 255, 225), 2)
        
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
