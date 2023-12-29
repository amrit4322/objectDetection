from flask import Flask , render_template,jsonify,request,send_file
from ultralytics import YOLO
from flask_socketio import SocketIO
import cv2
import base64
import threading
import os
# import urllib.request

app = Flask(__name__)
socketio = SocketIO(app)
model = YOLO('yolov8s.pt') 
stop_thread = False
thread_lock = threading.Lock()
# urllib.request.urlretrieve("https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true", "coco.names")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/objectDetectionFile', methods=['POST'])
def object_detection_file():
    # Handle file upload and perform object detection logic here
    file = request.files['file']
    
    if file:
        # Perform object detection logic on the uploaded file
        # You can replace the following line with your actual logic
        print("file ",file)

        file.save("uploads/"+file.filename)
        global stop_thread
        stop_thread=False
        socketio.start_background_task(target=object_detection_thread,source =f"uploads/{file.filename}",save=True)
        result = f"Object detection from file: {file.filename}"
        return jsonify({'result': result})
    else:
        return jsonify({'error': 'No file uploaded'})

@app.route('/objectDetectionWebCam')
def object_detection_webcam():
    global stop_thread
    stop_thread=False
    socketio.start_background_task(target=object_detection_thread,source=0)
    return "Ok",200

@app.route('/stopObjectDetection')
def stop_object_detection():
    global stop_thread
    with thread_lock:
        stop_thread = True
    return jsonify({'status': 'Stopping object detection thread'})

def object_detection_thread(source=0,save=False):
    global stop_thread
    cap = cv2.VideoCapture(source)
    if save==True:
        output_path =os.path.join("predictions", "result_video.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
       
    while stop_thread==False:
        ret, frame = cap.read()

        if not ret:
            break
        if save==False:
            frame = cv2.flip(frame, 1)
        # Perform YOLOv8 inference on the frame
        results = model.predict(frame)
        # print("results " ,results)
        # Draw bounding boxes on the frame
        for result in results:
            xyxys= result.boxes.xyxy
            class_id = result.boxes.cls
            confidences = result.boxes.conf
            for i,xyxy in enumerate(xyxys):
                if confidences[i]>0.5:
                    label = str(classes[int(class_id[i])])
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 3) 
                    cv2.putText(frame, f"{label}: {confidences[i]:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save==True:
            output_video.write(frame)
        # Encode the annotated frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer)

        # Send the base64-encoded frame to the client via WebSocket
        socketio.emit('update_frame', {'image': encoded_frame.decode('utf-8')})
        
        # Optionally, perform other real-time processing or visualization here
    
    cap.release()
    if save==True:
        output_video.release()
    socketio.emit("completed",{'value':'done'})
    cv2.destroyAllWindows()


@app.route('/download')
def download():
    # Assuming file_path is the path to the video file on the server
    # You might need to adjust the path accordingly
    file_path ="predictions/result_video.mp4"
    # Provide the correct MIME type for a video file (e.g., MP4)
    mime_type = 'video/mp4'
    print("donwloading")
    # Extract the filename from the file path
    filename = os.path.basename(file_path)
    # return filename
    # Send the video file for download
    return send_file(file_path, mimetype=mime_type, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)