import cv2
import pyzbar.pyzbar as pyzbar
import pyttsx3
import numpy as np

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Access the webcam video stream
webcam_video_stream = cv2.VideoCapture(0)

if not webcam_video_stream.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to exit the scanner.")

while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Detect barcodes and QR codes
    decoded_objects = pyzbar.decode(frame)
    if decoded_objects:
        for obj in decoded_objects:
            # Decode data and read aloud
            data = obj.data.decode("utf-8")
            print(f"Detected Data: {data}")
            engine.say(f"Detected data is: {data}")
            engine.runAndWait()

            # Draw rectangles around detected objects
            points = obj.polygon
            if points:
                pts = np.array([point for point in points], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the video feed
    cv2.imshow("Barcode and QR Code Scanner", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam_video_stream.release()
cv2.destroyAllWindows()
engine.stop()
