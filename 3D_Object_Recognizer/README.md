# 🧠 Real-Time Object Detection with Voice Interaction 🎙️

This project is a real-time intelligent vision assistant built using Python. It combines YOLOv8 object detection, speech recognition, OCR, and shape-color analysis to deliver interactive feedback via voice. It identifies objects from a webcam feed and speaks out details such as the object name, shape, color, and text found on it.

---

## 🚀 Features

- ✅ **Object Detection** using YOLOv8
- 🎨 **Color Identification** from dominant region pixels
- 🔺 **Shape Detection** via contour approximation
- 🔤 **Text Reading** using EasyOCR
- 🎙️ **Voice Commands** to interact with the system
- 🔊 **Speech Output** with detected object info
- 🎯 **Target Locking** to focus on one object at a time
- 🔁 **Change Target** via voice
- ❌ **Terminate Process** with voice

---

## 🎮 Voice Commands

| Voice Command   | Description                        |
|-----------------|------------------------------------|
| `what is this`  | System will describe the object    |
| `change target` | Switch focus to the next object    |
| `end process`   | Exit the program gracefully         |

---

## 🛠️ Requirements

You can install all dependencies via:

```bash
pip install -r requirements.txt
```

**Requirements:**

```
opencv-python
numpy
pyttsx3
speechrecognition
easyocr
ultralytics
torch
```

> Make sure your system supports microphone access and has a webcam.

---

## 🧪 Running the App

```bash
python main.py
```

Once the script starts:

- You'll see live webcam feed with detected object boxes.
- Say “what is this” to hear details about the locked object.
- Say “change target” to switch the detection focus.
- Say “end process” to stop execution.

---

## 📁 Project Structure

```
.
├── models/
│   └── yolov8n.pt          # YOLOv8 model file
├── main.py                 # Python script for detection
├── requirements.txt        # Python dependencies
└── README.md               # Project description
```

---

## 📷 Output Details

Each detection will include:

- ✅ Object Name (e.g., chair, book, etc.)
- 🔷 Shape: Triangle, Rectangle, Circle
- 🌈 Color: Red, Blue, Green, etc.
- 🧾 Detected Text: If text is present
- 🎯 Confidence: Accuracy score from YOLO

Example:

```
Chair, color blue, shape rectangle, text "Ergo", accuracy 87.53%
```

---

## 📌 Libraries Used

- [`ultralytics`](https://github.com/ultralytics/ultralytics) – YOLOv8 object detection
- [`OpenCV`](https://opencv.org/) – Image processing and webcam handling
- [`EasyOCR`](https://github.com/JaidedAI/EasyOCR) – For reading text from detected objects
- [`SpeechRecognition`](https://pypi.org/project/SpeechRecognition/) – Voice command recognition
- [`pyttsx3`](https://pypi.org/project/pyttsx3/) – Text-to-speech output
- [`NumPy`](https://numpy.org/) – Array math for color detection

---

## 📸 Screenshots

_Add sample screenshots or GIFs of the system running, showing object detection and voice responses._

---

## ✍️ Author

**Vignesh**  
`Agent Kill` | B.Tech CSE (AI & DS)

---

## 📃 License

This project is licensed under the MIT License. Feel free to modify and use it in your own work.

---

Enjoy building with AI! 💡
