// Connect to the WebSocket server
var socket = io.connect('http://' + document.domain + ':' + location.port);

// Listen for updates from the server to update the webcam feed
socket.on('update_frame', function (data) {
    // Display the webcam feed and update the live-stream image source
    document.getElementById('webcamFeed').style.display = "block";
    document.getElementById('live-stream').src = 'data:image/jpeg;base64,' + data.image;
    document.getElementById('container').style.display = "block";
});

// Listen for completion event from the server
socket.on('completed', function (data) {
    if (data.value == "done") {
        // Display the download button and hide the webcam feed and container
        document.getElementById('downloadbtnid').style.display = "block";
        document.getElementById('webcamFeed').style.display = "none";
        document.getElementById('container').style.display = "none";
    }
});

// Listen for completion event from the server (for webcam capture)
socket.on('completed_web', function (data) {
    if (data.value == "done") {
        // Hide the webcam feed and container
        document.getElementById('webcamFeed').style.display = "none";
        document.getElementById('container').style.display = "none";
    }
});

// Function to capture video from webcam and initiate object detection
function captureVideo() {
    document.getElementById('downloadbtnid').style.display = "none";
    document.getElementById('container').style.display = "block";

    // Set up canvas and video elements
    var canvas = document.getElementById('canvas');
    var context = canvas.getContext('2d');
    const video = document.querySelector("#videoElement");

    video.width = 600;
    video.height = 550;

    // Access webcam and start video stream
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
                video.play();
                console.log("Capturing video");
            })
            .catch(function (error) {
                console.log("Error:", error);
            });
    }

    // Periodically capture frames and send to the server
    const FPS = 6;
    setInterval(() => {
        width = video.width;
        height = video.height;
        context.drawImage(video, 0, 0, width, height);
        var data = canvas.toDataURL('image/jpeg');
        context.clearRect(0, 0, width, height);
        socket.emit('capture_frame', data);
    }, 1000 / FPS);
}



// Function to stop object detection and hide the webcam feed
function stopObjectDetection() {
    console.log('Object detection stopping');
    // Delay the hiding of the webcam feed for better visual feedback
    setTimeout(function () {
        document.getElementById('webcamFeed').style.display = "none";
    }, 1000);
}

// Function to handle the download button click
function downloadbtn() {
    console.log('Downloading the file');
}
