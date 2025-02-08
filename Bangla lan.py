import threading
from ultralytics import YOLO
import cv2
from googletrans import Translator
from gtts import gTTS
import os
import time

# Initialize translator
translator = Translator()

# Lock for threading
announcement_lock = threading.Lock()

# Function to announce detected objects in the selected language
def announce_message(message, lang):
    with announcement_lock:
        if lang == "en":
            tts = gTTS(text=message, lang="en")
        else:
            translated_message = translator.translate(message, src="en", dest=lang).text
            tts = gTTS(text=translated_message, lang=lang)
        tts.save("announcement.mp3")
        os.system("start announcement.mp3")  # Use 'afplay' on macOS or 'mpg123' on Linux
        time.sleep(2)  # Delay to allow the announcement to finish

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Update with the correct model path if necessary

# Access webcam video stream
webcam_video_stream = cv2.VideoCapture(0)

last_announced_objects = set()  # Track the last announced objects
announcement_interval = 5  # Minimum time (seconds) between announcements
last_announcement_time = time.time()

print("Press 'q' to exit.")

while True:
    # Capture a frame from the webcam
    ret, frame = webcam_video_stream.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    # Run YOLOv8 prediction
    results = model.predict(source=frame, save=False, conf=0.25, verbose=False)

    detected_objects = []
    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        confidences = result.boxes.conf

        # Process only top 1–2 objects based on confidence
        top_objects = sorted(zip(boxes, classes, confidences), key=lambda x: x[2], reverse=True)[:2]

        for box, cls, confidence in top_objects:
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            detected_objects.append(class_name)

            # Draw bounding box and label
            box_color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Announce detected objects at regular intervals
    current_time = time.time()
    if current_time - last_announcement_time > announcement_interval:
        unique_objects = set(detected_objects)
        new_objects = unique_objects - last_announced_objects

        if new_objects:
            # Announce up to 1 new object
            objects_to_announce = list(new_objects)[:1]
            for obj in objects_to_announce:
                threading.Thread(target=announce_message, args=(f"I see a {obj}.", "bn"), daemon=True).start()
            last_announced_objects = unique_objects
        else:
            threading.Thread(target=announce_message, args=("Clear path ahead.", "bn"), daemon=True).start()

        last_announcement_time = current_time

    # Display the detection output
    cv2.imshow("Detection Output", frame)

    # Terminate if '
