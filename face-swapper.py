import cv2
import dlib
import numpy as np

def extract_landmarks(face):
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    landmarks = predictor(gray, face)
    landmark_points = []

    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmark_points.append((x, y))

    return np.array(landmark_points, np.int32)


cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)

    faces = detector(gray)

    if len(faces) == 2:
        face1_landmarks = extract_landmarks(faces[0])
        face2_landmarks = extract_landmarks(faces[1])

        #faces_landmarks = np.add(face1_landmarks, face2_landmarks)
        hull1 = cv2.convexHull(face1_landmarks)
        hull2 = cv2.convexHull(face2_landmarks)

        #cv2.polylines(frame, [hull1], True, (0, 255, 0), 1)
        #cv2.polylines(frame, [hull2], True, (0, 255, 0), 1)
        
        cv2.fillConvexPoly(mask, hull1, 255)
        cv2.fillConvexPoly(mask, hull2, 255)

        faces = cv2.bitwise_and(frame, frame, mask = mask)

        cv2.imshow("Face swap", faces)

    else:
        cv2.imshow("Face swap", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break