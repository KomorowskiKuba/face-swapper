import cv2
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    if len(faces) == 2:
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()

            landmarks = predictor(gray, face)

            for i in range(0, 68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)




    cv2.imshow("Face swap", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break