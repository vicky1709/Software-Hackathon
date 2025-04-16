import cv2
import numpy as np
import pyttsx3
import easyocr
from ultralytics import YOLO
import speech_recognition as sr
import threading

# Define a set of color names with their RGB values
COLOR_LABELS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "brown": (165, 42, 42),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203)
}

# Initialize YOLO object detection model
yolo_model = YOLO('models/yolov8n.pt')

# Initialize EasyOCR for text detection
ocr_reader = easyocr.Reader(['en'])

# Setup for speech engine
speech_engine = pyttsx3.init()

# Initialize speech recognition
speech_recognizer = sr.Recognizer()

# Flags to manage program flow
should_speak = False
should_stop = False
current_target_index = 0
target_change_flag = False
detection_data = None
person_class_id = 0  # ID for person class

# Function to identify the closest matching color name
def find_closest_color(rgb):
    min_distance = float('inf')
    closest_color_name = None
    for color_name, color_rgb in COLOR_LABELS.items():
        distance = np.linalg.norm(np.array(rgb) - np.array(color_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name
    return closest_color_name

# Function to extract the dominant color of a given image region
def extract_dominant_color(image):
    image_resized = cv2.resize(image, (50, 50))  # Reduce image size for speed
    pixel_data = image_resized.reshape(-1, 3)
    avg_color = np.mean(pixel_data, axis=0)
    blue, green, red = avg_color
    return find_closest_color((int(red), int(green), int(blue)))

# Function to detect the shape of an object based on contours
def detect_object_shape(cropped_image):
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            return 'Triangle'
        elif len(approx) == 4:
            return 'Rectangle'
        elif len(approx) > 4:
            return 'Circle'
    return 'Unknown'

# Function to speak out detected information
def speak_out(text):
    def _speak():
        print("[Speech Output]:", text)
        speech_engine.say(text)
        speech_engine.runAndWait()
    
    threading.Thread(target=_speak).start()

# Function to listen for voice commands
def listen_for_commands():
    global should_speak, should_stop, target_change_flag
    while True:
        with sr.Microphone() as source:
            print("[Voice Command] Listening...")
            audio_data = speech_recognizer.listen(source)
            try:
                command = speech_recognizer.recognize_google(audio_data).lower()
                print(f"[Command Received]: {command}")
                
                if "what is this" in command:
                    should_speak = True
                elif "end process" in command:
                    should_stop = True
                    speak_out("Terminating the process.")
                    break
                elif "change target" in command:
                    target_change_flag = True
                    speak_out("Target has been changed.")
            except:
                continue

# Start listening for commands in a separate thread
threading.Thread(target=listen_for_commands, daemon=True).start()

# Initialize webcam feed
camera = cv2.VideoCapture(0)

# Set frame resolution for the webcam
frame_width = 1280
frame_height = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

while True:
    success, frame = camera.read()
    if not success or should_stop:
        break

    # Run object detection on the captured frame
    detections = yolo_model(frame)[0]
    processed_frame = frame.copy()

    detection_data = None
    bounding_boxes = detections.boxes

    if bounding_boxes and len(bounding_boxes) > 0:
        # Prioritize non-person objects
        sorted_boxes = sorted(bounding_boxes, key=lambda box: 1 if int(box.cls[0]) == person_class_id else 0)

        if target_change_flag:
            # Change target object when triggered
            current_target_index = (current_target_index + 1) % len(sorted_boxes)
            target_change_flag = False

        if current_target_index < len(sorted_boxes):
            box = sorted_boxes[current_target_index]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            confidence = float(box.conf[0]) * 100

            # Crop the object region from the frame for color and shape detection
            cropped_object = frame[y1:y2, x1:x2]
            dominant_color = extract_dominant_color(cropped_object)
            object_shape = detect_object_shape(cropped_object)
            text_results = ocr_reader.readtext(cropped_object)
            detected_text = text_results[0][-2] if text_results else "No text detected"

            # Prepare detection result description
            detection_data = f"{label}, color {dominant_color}, shape {object_shape}, text {detected_text}, accuracy {confidence:.2f}%"

            # Draw bounding box around the object
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            info_text = f"{label} | {object_shape} | {dominant_color} | {confidence:.2f}%"
            cv2.putText(processed_frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the processed frame with detections
    cv2.imshow("Live Object Detection", processed_frame)

    # Speak out the detection result if needed
    if should_speak:
        should_speak = False
        speak_out(detection_data if detection_data else "No objects detected.")

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
camera.release()
cv2.destroyAllWindows()
