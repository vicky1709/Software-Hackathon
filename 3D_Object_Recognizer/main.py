import cv2
import numpy as np
import pyttsx3
import easyocr
from ultralytics import YOLO
import speech_recognition as sr
import threading
import time
from queue import Queue

# Load YOLO model
model = YOLO('models/yolov8n.pt')

# EasyOCR reader
reader = easyocr.Reader(['en'])

# Text-to-speech engine
engine = pyttsx3.init()

# Speech recognizer
recognizer = sr.Recognizer()

# Global flags
speak_result = False
stop_program = False
detection_result = None
change_target = False
locked_target = None
lock_threshold = 100  # pixels
person_class_id = 0  # Adjust if different for your model

# Create a queue for passing OCR results from the thread
ocr_queue = Queue()

def detect_shape(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        sides = len(approx)
        if sides == 3: return 'Triangle'
        elif sides == 4: return 'Rectangle'
        elif sides == 5: return 'Pentagon'
        elif sides == 6: return 'Hexagon'
        elif sides == 10: return 'Star'
        elif sides > 10: return 'Circle'
    return 'Unknown'

def speak(text):
    def _speak():
        print("[Speech]:", text)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

def listen_commands():
    global speak_result, stop_program, change_target
    while True:
        with sr.Microphone() as source:
            print("[Voice] Listening...")
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"[Command]: {command}")
                if "what is this" in command:
                    speak_result = True
                elif "end process" in command:
                    stop_program = True
                    speak("Okay, stopping the program.")
                    break
                elif "change target" in command:
                    change_target = True
                    speak("Changing target.")
            except:
                continue

# Start listening thread
threading.Thread(target=listen_commands, daemon=True).start()

# Speak startup message
speak("deepscope Assistant is now running. Say 'what is this' to describe an object or 'end process' to stop.")

# Function to run OCR in a separate thread
def ocr_thread(cropped_image):
    # Resizing and preprocessing for OCR
    if cropped_image.shape[0] > 20 and cropped_image.shape[1] > 20:
        resized_crop = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
        text_results = reader.readtext(gray_crop)

        # Improved OCR handling
        if text_results:
            detected_text = ' '.join([result[-2] for result in text_results])
        else:
            detected_text = "No readable text found"
    else:
        detected_text = "Too small for text"
    
    # Push the result to the queue
    ocr_queue.put(detected_text)

# Open webcam
cap = cv2.VideoCapture(0)
frame_width = 1920
frame_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# FPS setup
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret or stop_program:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    results = model(frame)[0]
    annotated_frame = frame.copy()
    detection_result = None
    boxes = results.boxes

    closest_box = None
    min_distance = float('inf')

    if boxes and len(boxes) > 0:
        if change_target or locked_target is None:
            sorted_boxes = sorted(boxes, key=lambda box: 1 if int(box.cls[0]) == person_class_id else 0)
            if sorted_boxes:
                box = sorted_boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                locked_target = {
                    'class_id': class_id,
                    'center': center
                }
                change_target = False

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if locked_target and class_id == locked_target['class_id']:
                dist = np.linalg.norm(np.array(locked_target['center']) - np.array((cx, cy)))
                if dist < min_distance and dist < lock_threshold:
                    closest_box = (box, (x1, y1, x2, y2), (cx, cy))
                    min_distance = dist

        if closest_box:
            box, (x1, y1, x2, y2), center = closest_box
            locked_target['center'] = center
            class_id = int(box.cls[0])
            label = model.names[class_id]
            conf = float(box.conf[0]) * 100
            cropped = frame[y1:y2, x1:x2]

            # Detect shape
            shape = detect_shape(cropped)

            # Start OCR in a separate thread
            threading.Thread(target=ocr_thread, args=(cropped,), daemon=True).start()

            # Wait for the OCR result (this is a blocking call)
            detected_text = ocr_queue.get()

            detection_result = f"{label}, shape {shape}, text {detected_text}, accuracy {conf:.2f}%"

            # Draw detection box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            info = f"{label} | {shape} | {conf:.2f}%"
            cv2.putText(annotated_frame, info, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw FPS
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Object Detection with Shape & Text", annotated_frame)

    if speak_result:
        speak_result = False
        speak(detection_result if detection_result else "No objects detected.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
