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

        rectangle1 = cv2.boundingRect(hull1)
        #(x1, y1, w1, h1) = rectangle1
        #cv2.rectangle(faces, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0))

        rectangle2 = cv2.boundingRect(hull2)
        #(x2, y2, w2, h2) = rectangle2
        #cv2.rectangle(faces, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0))

        subdiv_1 = cv2.Subdiv2D(rectangle1)
        subdiv_1.insert(face1_landmarks.tolist())
        triangles_1 = subdiv_1.getTriangleList()
        triangles_1 = np.array(triangles_1, dtype=np.int32)
        #print(triangles_1)

        subdiv_2 = cv2.Subdiv2D(rectangle2)
        subdiv_2.insert(face2_landmarks.tolist())
        triangles_2 = subdiv_2.getTriangleList()
        triangles_2 = np.array(triangles_2, dtype=np.int32)

        for t in triangles_1:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.line(frame, pt2, pt3, (0, 0, 255), 2)
            cv2.line(frame, pt1, pt3, (0, 0, 255), 2)

        for t in triangles_2:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.line(frame, pt2, pt3, (0, 0, 255), 2)
            cv2.line(frame, pt1, pt3, (0, 0, 255), 2)
        #print(triangles_2)



        cv2.imshow("Face swap", frame)

    else:
        cv2.imshow("Face swap", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break