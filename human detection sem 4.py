# for computer vision tasks
import cv2
# for numerical operation
import numpy as np

cap = cv2.VideoCapture('pexels_videos_2670 (1080p).mp4')

human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Variables for object tracking
objects = {}  # Dictionary to store object IDs and centroids

# Specify the desired width and height for the resized frame
width = 1000
height = 600

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame to the specified width and height
    frame = cv2.resize(frame, (width, height))

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans
    scaleFactor = 1.1
    minNeighbors = 5
    minSize = (30, 30)  # Minimum size of the detected object
    maxSize = (200, 200)  # Maximum size of the detected object
    humans = human_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,
                                            minSize=minSize, maxSize=maxSize)

    # Process detected humans and track them
    for (x, y, w, h) in humans:
        centroid_x = x + (w // 2)
        centroid_y = y + (h // 2)

        # Check if the detected human is already being tracked
        found_match = False
        for obj_id, centroid in objects.items():
            prev_centroid_x, prev_centroid_y = centroid

            # Compute the Euclidean distance between the current and previous centroids
            distance = np.sqrt((centroid_x - prev_centroid_x) ** 2 + (centroid_y - prev_centroid_y) ** 2)

            # If the distance is smaller than a threshold, assume it's the same object and update the centroid
            if distance < 50:
                objects[obj_id] = (centroid_x, centroid_y)
                found_match = True
                break

        # If no match is found, assign a new ID to the object and start tracking
        if not found_match:
            obj_id = len(objects)
            objects[obj_id] = (centroid_x, centroid_y)

        # Draw rectangle and ID label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f'Human {obj_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
