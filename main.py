import streamlit as st
import cv2
import numpy as np
import os

# Function to detect faces and draw bounding boxes
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Function to detect age and gender with improved text display
def detectAgeGender(frame, faceNet, ageNet, genderNet, MODEL_MEAN_VALUES, ageList, genderList, conf_threshold, padding=20):
    resultImg, faceBoxes = highlightFace(faceNet, frame, conf_threshold)
    if not faceBoxes:
        return resultImg, "No face detected"

    predictions = []
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Age prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        
        label = f"{gender}, {age}"
        predictions.append(label)
        
        # Calculate dynamic font size based on box height
        box_height = faceBox[3] - faceBox[1]
        font_scale = max(0.5, min(1.2, box_height / 100))  # Scale between 0.5 and 1.2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        
        # Adjust text position to stay above the box and within frame
        text_x = faceBox[0]
        text_y = max(faceBox[1] - text_height - 5, 10)  # Ensure it stays above box and within frame (min y = 10)
        
        # Draw a filled rectangle as text background for better contrast
        bg_top_left = (text_x, text_y - text_height - baseline)
        bg_bottom_right = (text_x + text_width, text_y + baseline)
        cv2.rectangle(resultImg, bg_top_left, bg_bottom_right, (0, 0, 0), cv2.FILLED)  # Black background
        
        # Draw text in white for high contrast
        cv2.putText(resultImg, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    
    return resultImg, predictions

# Model paths
faceProto = r"C:\\Users\\swag\\OneDrive\\Desktop\\pbl\\Gender-and-Age-Detection\\opencv_face_detector.pbtxt"
faceModel = r"C:\\Users\\swag\\OneDrive\\Desktop\\pbl\\Gender-and-Age-Detection\\opencv_face_detector_uint8.pb"
ageProto = r"C:\\Users\\swag\\OneDrive\\Desktop\\pbl\\Gender-and-Age-Detection\\age_deploy.prototxt"
ageModel = r"C:\\Users\\swag\\OneDrive\\Desktop\\pbl\\Gender-and-Age-Detection\\age_net.caffemodel"
genderProto = r"C:\\Users\\swag\\OneDrive\\Desktop\\pbl\\Gender-and-Age-Detection\\gender_deploy.prototxt"
genderModel = r"C:\\Users\\swag\\OneDrive\\Desktop\\pbl\\Gender-and-Age-Detection\\gender_net.caffemodel"

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(20-28)', '(30-40)', '(40-50)', '(50-100)']
genderList = ['Male', 'Female']

# Streamlit GUI
st.set_page_config(page_title="Age & Gender Detector", layout="wide")
st.title("🎉 Age and Gender Detector")
st.markdown("Detect age and gender from images, videos, or your webcam with ease!")

# Input method selection on main page
st.header("Choose Your Input Method")
input_method = st.radio("Select an option:", ("Upload Image", "Use Webcam", "Upload Video"), horizontal=True)

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.05)
    st.markdown("---")
    st.info("Adjust the confidence threshold to fine-tune face detection sensitivity.")

# Main content
if input_method == "Upload Image":
    st.header("Upload an Image")
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, JPEG, PNG")
    
    with col2:
        st.write("")  # Spacer
        process_button = st.button("Process Image", use_container_width=True, type="primary")

    if uploaded_file and process_button:
        with st.spinner("Processing your image..."):
            # Convert the file to an OpenCV image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            # Process the image
            result_img, predictions = detectAgeGender(image, faceNet, ageNet, genderNet, MODEL_MEAN_VALUES, ageList, genderList, conf_threshold)
            
            # Convert BGR to RGB for display
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            # Display results
            st.image(result_img_rgb, caption="Processed Image", use_container_width=True)
            if isinstance(predictions, str):
                st.warning(predictions)
            else:
                st.success("Predictions:")
                for pred in predictions:
                    st.write(f"• {pred}")

elif input_method == "Use Webcam":
    st.header("Live Webcam Detection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        FRAME_WINDOW = st.image([], use_container_width=True)
    
    with col2:
        st.write("")  # Spacer
        start_button = st.button("Start Webcam", use_container_width=True, type="primary", key="start_webcam")
        stop_button = st.button("Stop Webcam", use_container_width=True, key="stop_webcam")  # Always enabled
    
    # Initialize session state
    if "run_webcam" not in st.session_state:
        st.session_state.run_webcam = False
    
    # Start webcam if button is pressed
    if start_button:
        st.session_state.run_webcam = True
    
    # Stop webcam if button is pressed
    if stop_button:
        st.session_state.run_webcam = False
    
    # Webcam loop with DirectShow backend
    if st.session_state.run_webcam:
        # Use DirectShow backend (CAP_DSHOW) instead of MSMF
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error("Could not access webcam. Ensure it’s connected and not in use by another application.")
            st.session_state.run_webcam = False
        else:
            with st.spinner("Streaming webcam..."):
                while st.session_state.run_webcam and "run_webcam" in st.session_state and st.session_state.run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video. Webcam may be disconnected or unavailable.")
                        st.session_state.run_webcam = False
                        break
                    
                    # Process the frame
                    result_img, _ = detectAgeGender(frame, faceNet, ageNet, genderNet, MODEL_MEAN_VALUES, ageList, genderList, conf_threshold)
                    
                    # Convert BGR to RGB
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    FRAME_WINDOW.image(result_img_rgb, use_container_width=True)
                
                cap.release()
                st.session_state.run_webcam = False
                st.success("Webcam stopped.")

elif input_method == "Upload Video":
    st.header("Upload a Video")
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_video = st.file_uploader("Drop a video here", type=["mp4", "avi"], help="Supported formats: MP4, AVI")
    
    with col2:
        st.write("")  # Spacer
        process_button = st.button("Process Video", use_container_width=True, type="primary")

    if uploaded_video and process_button:
        with st.spinner("Processing your video..."):
            # Save uploaded video to a temporary file
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(uploaded_video.read())
            
            cap = cv2.VideoCapture(temp_file)
            FRAME_WINDOW = st.image([], use_container_width=True)
            
            if not cap.isOpened():
                st.error("Could not open video file.")
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process the frame
                    result_img, _ = detectAgeGender(frame, faceNet, ageNet, genderNet, MODEL_MEAN_VALUES, ageList, genderList, conf_threshold)
                    
                    # Convert BGR to RGB
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    # Display frame
                    FRAME_WINDOW.image(result_img_rgb, use_container_width=True)
                
                cap.release()
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                st.success("Video processing complete.")

# Footer
st.markdown("---")
st.write("Powered by **Streamlit** and **OpenCV** | Built with ❤️ by Satyabrat Panda")
