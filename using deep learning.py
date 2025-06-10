from mtcnn import MTCNN
import cv2

# Initialize face detector
detector = MTCNN()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (MTCNN uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector.detect_faces(frame_rgb)

    # Draw bounding boxes and count
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display count
    cv2.putText(frame, f'Faces: {len(faces)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("MTCNN Face Counter - Press 'q' to Quit", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
