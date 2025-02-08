import threading
from ultralytics import YOLO
import cv2
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Function to announce detected objects with distance and direction
def announce_message(message):
    engine.say(message)
    engine.runAndWait()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Access webcam video stream
webcam_video_stream = cv2.VideoCapture(0)

# Focal length and real-world object width (in cm)
focal_length = 700  # Approximate value; adjust based on your camera
real_width = 50  # Default real-world width for a person (in cm)

while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    results = model.predict(source=frame, save=False, conf=0.25, verbose=False, device="cpu")

    no_objects_detected = True  # Flag to determine if no objects are detected

    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        confidences = result.boxes.conf

        for box, cls, confidence in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            confidence = float(confidence)

            width_in_pixels = x2 - x1
            if width_in_pixels > 0:
                distance = (real_width * focal_length) / width_in_pixels
                distance_in_meters = distance / 100  # Convert to meters

                # Determine object direction
                center_x = (x1 + x2) // 2
                if center_x < frame_width // 3:
                    direction = "left"
                elif center_x > 2 * frame_width // 3:
                    direction = "right"
                else:
                    direction = "center"

                no_objects_detected = False  # Objects are detected

                # Announce object with distance and direction
                threading.Thread(
                    target=announce_message,
                    args=(f"I see a {class_name} at approximately {distance_in_meters:.2f} meters to your {direction}.",),
                    daemon=True,
                ).start()

                # Draw bounding box and label
                box_color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f} Dist: {distance_in_meters:.2f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    2,
                )

    # Announce clear path if no objects are detected
    if no_objects_detected:
        threading.Thread(
            target=announce_message,
            args=("Clear path ahead, you can move forward.",),
            daemon=True,
        ).start()

    # Display the detection output
    cv2.imshow("Detection Output", frame)

    # Terminate if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV windows
webcam_video_stream.release()
cv2.destroyAllWindows()
