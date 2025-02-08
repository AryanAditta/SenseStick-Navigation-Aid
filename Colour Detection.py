import cv2
import numpy as np
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Define HSV color ranges for Red, Green, Blue, White, Black
color_ranges = {
    "Red": [(0, 120, 70), (10, 255, 255)],      # Red lower range
    "Green": [(36, 50, 70), (89, 255, 255)],    # Green range
    "Blue": [(90, 50, 70), (128, 255, 255)],    # Blue range
    "White": [(0, 0, 200), (180, 20, 255)],     # White range
    "Black": [(0, 0, 0), (180, 255, 50)]        # Black range
}

def get_color_name(h, s, v):
    """
    Determine the color name based on HSV values.
    """
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
            return color_name
    return "Unknown"

# Initialize webcam
webcam_video_stream = cv2.VideoCapture(0)

if not webcam_video_stream.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to exit.")

last_color_name = None  # Track the last announced color

while True:
    ret, frame = webcam_video_stream.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize the frame for better performance
    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the center region of the frame
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    center_region = hsv_frame[center_y - 10:center_y + 10, center_x - 10:center_x + 10]

    # Calculate the mean color in the center region
    mean_hsv = np.mean(center_region, axis=(0, 1))
    h, s, v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

    # Get the color name
    color_name = get_color_name(h, s, v)

    # Announce the detected color if it's different from the last one
    if color_name != last_color_name:
        print(f"Detected Color: {color_name}")
        engine.say(f"The color is {color_name}")
        engine.runAndWait()
        last_color_name = color_name

    # Draw a rectangle and display the color name
    cv2.rectangle(frame, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (255, 255, 255), 2)
    cv2.putText(frame, f"Color: {color_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Color Detection with HSV", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam_video_stream.release()
cv2.destroyAllWindows()
