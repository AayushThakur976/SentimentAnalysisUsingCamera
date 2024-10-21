import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

# Load the model from JSON
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load weights
model.load_weights("facialemotionmodel.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

frame_count = 0  # Initialize frame counter

while True:
    ret, im = webcam.read()
    if not ret:
        break  # Exit if the frame is not captured

    frame_count += 1
    if frame_count % 2 == 0:  # Process every second frame
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

        if len(faces) > 0:
            for (p, q, r, s) in faces:
                cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)  # Draw rectangle around the face
                image = gray[q:q+s, p:p+r]
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]

                # Debugging: Print prediction probabilities
                print("Prediction probabilities:", pred)
                print("Predicted label:", prediction_label)

                cv2.putText(im, prediction_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    cv2.imshow("Output", im)  # Show the output video

    # Break the loop on 'q' key press
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
