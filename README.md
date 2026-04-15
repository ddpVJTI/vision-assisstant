# 🎯 Smart Vision Assistant

Welcome to the **Smart Vision Assistant** project! 

This application was built from the ground up to serve as a real-time smart object identifier specifically designed to help visually impaired individuals. It uses a webcam, artificial intelligence, and audio narration to describe the world dynamically!

---

## What Does This Actually Do?

Instead of just looking at camera feeds blindly, this app:
1. **Sees what's in front of you:** Using a highly optimized, pre-trained AI model (`YOLOv8`), it identifies 80 common objects (like people, chairs, bottles, or cars).
2. **Understands where things are:** It instantly figures out if the object is to your **Left**, **Center**, or **Right**.
3. **Guesses how far it is:** By analyzing how large the object is on the screen, it warns you if something is **Very Close**, **Near**, or **Far**.
4. **Speaks to you:** It pieces this all together into helpful, spoken audio snippets like *"Person on the left, near"* so you know exactly what is happening without looking at the screen.
5. **Doesn't annoy you!** It has a built-in "cooldown" so it won't repeatedly yell *"Chair in center"* 50 times a second if you're just standing still.

---

## 📁 How is this Project Organized?

I've structured this repository to be clean, balanced, and easy to navigate:

```text
smart-vision-assistant/
│
├── docs/                     # Human-readable documentation 
│   └── IMPLEMENTATION_PLAN.md # Read this for the deep-dive on how the code actually works!
│
├── scripts/                  # Helper tools
│   └── download_model.py     # Script to snag the AI weights from the internet
│
├── src/                      # The actual meat of the application (Source Code)
│   ├── streamlit_app.py      # The beautiful web dashboard you interact with
│   ├── detector.py           # The smart brain that processes the camera and YOLO
│   ├── audio_engine.py       # The background worker that speaks text out loud
│   └── config.py             # Settings you can tweak (like distance thresholds)
│
├── weights/                  # Where the heavy AI model lives
│   └── yolov8n.pt            # The ~6MB pretrained neural network file
│
├── requirements.txt          # A list of external libraries required to make it run
└── README.md                 # You are reading this right now!
```

---

## 🚀 How Do I Use It?

### 1. Install Everything
Assuming you already have Python installed, crack open your terminal and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
Start up the web dashboard (this will pop open in your browser automatically):
```bash
python -m streamlit run src/streamlit_app.py
```

### 3. Start Inferencing!
You have three options in the sidebar:
* **Live Camera**: Uses your webcam for real-time tracking.
* **Upload Photo**: Test the system with any static image file.
* **Upload Video**: Upload a pre-recorded `.mp4` and have the AI narrate what happens!

*(Pro tip: Turn on the **"⚠ Only Alert on Danger Objects"** filter to ignore harmless items like chairs/bottles and only get notified of hazards!)*
