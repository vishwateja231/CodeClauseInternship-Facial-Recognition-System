from tkinter import filedialog
import face_recognition
import cv2
import os

# Load Haar cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load faces from the database
directory_path = 'database/'
faces_dict = {}
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        person_name = os.path.splitext(filename)[0]
        file_path = os.path.join(directory_path, filename)
        faces_dict[person_name] = file_path

# Select an image
file_path = filedialog.askopenfilename()
while not file_path:
    file_path = filedialog.askopenfilename()

# Load and process the input image
live = cv2.imread(file_path)
gray = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 2
count = 0

# Iterate through stored faces
for person_name, image_path in faces_dict.items():
    first_image = face_recognition.load_image_file(image_path)
    try:
        first_encoding = face_recognition.face_encodings(first_image)[0]
    except IndexError:
        print(f'Face not detected in database image: {person_name}')
        continue

    if len(faces) == 0:
        print("No faces found in the input image.")
    else:
        for (x, y, w, h) in faces:
            face = live[y:y+h, x:x+w]
            face_path = 'face.jpg'
            cv2.imwrite(face_path, face)

            try:
                face_img = face_recognition.load_image_file(face_path)
                second_encoding = face_recognition.face_encodings(face_img)[0]
                os.remove(face_path)  # Clean up temporary file

                # Compare face encodings
                result = face_recognition.compare_faces([first_encoding], second_encoding)
                if result[0]:
                    count += 1
                    cv2.rectangle(live, (x, y), (x + w, y + h), (127, 0, 255), 2)

                    # Get text size
                    text_size = cv2.getTextSize(person_name, font, font_scale, thickness)[0]
                    bg_x, bg_y = x, y - text_size[1] - 5

                    # Draw a filled rectangle as background for text
                    cv2.rectangle(live, (bg_x, bg_y), (bg_x + text_size[0], bg_y + text_size[1] + 5), (0, 0, 0), -1)
                    cv2.putText(live, person_name, (x, y - 10), font, font_scale, font_color, thickness)

            except IndexError:
                print(f'Face not detected in the selected image for {person_name}')
            except Exception as e:
                print(f'Error processing {person_name}: {e}')

if count == 0:
    print("No face was recognized.")
elif count == 1:
    print("1 face was recognized.")
else:
    print(f'{count} faces were recognized.')

cv2.imwrite("result.jpg", live)
