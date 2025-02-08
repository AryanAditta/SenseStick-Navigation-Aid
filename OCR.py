import cv2
import numpy as np
from paddleocr import PaddleOCR
import pyttsx3

# Initialize PaddleOCR with English model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Access the webcam
webcam_video_stream = cv2.VideoCapture(0)

while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert to RGB for PaddleOCR compatibility
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform OCR
    results = ocr.ocr(rgb_frame)

    # Check if results are valid
    if results and results[0]:
        for result in results[0]:
            text, confidence = result[1]
            print(f"Detected Text: {text}, Confidence: {confidence:.2f}")

            # Read the text aloud
            engine.say(f": {text}")
            engine.runAndWait()

            # Draw detection boxes and text on the frame
            points = np.array(result[0]).astype(np.int32)
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = points[0]  # Top-left corner of the bounding box
            cv2.putText(frame, f"{text} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        print("No text detected in the current frame.")

    # Display the frame
    cv2.imshow("Text Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam_video_stream.release()
cv2.destroyAllWindows()
