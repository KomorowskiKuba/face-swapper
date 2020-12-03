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

        subdiv1 = cv2.Subdiv2D(rectangle1)
        subdiv1.insert(face1_landmarks.tolist())
        triangles1 = subdiv1.getTriangleList()
        triangles1 = np.array(triangles1, dtype=np.int32)
        #print(triangles_1)

        subdiv2 = cv2.Subdiv2D(rectangle2)
        subdiv2.insert(face2_landmarks.tolist())
        triangles2 = subdiv2.getTriangleList()
        triangles2 = np.array(triangles2, dtype=np.int32)

        for t1, t2 in zip(triangles1, triangles2):
            pt1_t1 = (t1[0], t1[1])
            pt2_t1 = (t1[2], t1[3])
            pt3_t1 = (t1[4], t1[5])

            cv2.line(frame, pt1_t1, pt2_t1, (0, 0, 255), 2)
            cv2.line(frame, pt2_t1, pt3_t1, (0, 0, 255), 2)
            cv2.line(frame, pt1_t1, pt3_t1, (0, 0, 255), 2)

            pt1_t2 = (t2[0], t2[1])
            pt2_t2 = (t2[2], t2[3])
            pt3_t2 = (t2[4], t2[5])

            cv2.line(frame, pt1_t2, pt2_t2, (0, 0, 255), 2)
            cv2.line(frame, pt2_t2, pt3_t2, (0, 0, 255), 2)
            cv2.line(frame, pt1_t2, pt3_t2, (0, 0, 255), 2)

            triangle1 = np.array([pt1_t1, pt2_t1, pt3_t1], dtype=np.int32)
            rect1 = cv2.boundingRect(triangle1)
            (x1, y1, w1, h1) = rect1
            cropped_triangle1 = frame[y1: y1 + h1, x1: x1 + w1]
            cropped_triangle1_mask = np.zeros_like(cropped_triangle1)
            points1 = np.array([[pt1_t1[0] - x1, pt1_t1[1] - y1],
                                [pt2_t1[0] - x1, pt2_t1[1] - y1],
                                [pt3_t1[0] - x1, pt3_t1[1] - y1]], dtype=np.int32)
            cv2.fillConvexPoly(cropped_triangle1_mask, points1, (255, 0, 0))
            #cropped_triangle_1 = cv2.bitwise_and(cropped_triangle1, cropped_triangle1, mask=cropped_triangle1_mask)

            points1 = np.float32(points1)

            triangle2 = np.array([pt1_t2, pt2_t2, pt3_t2], dtype=np.int32)
            rect2 = cv2.boundingRect(triangle2)
            (x2, y2, w2, h2) = rect2
            cropped_triangle2 = frame[y2: y2 + h2, x2: x2 + w2]
            cropped_triangle2_mask = np.zeros_like(cropped_triangle2)
            points2 = np.array([[pt1_t2[0] - x2, pt1_t2[1] - y2],
                                [pt2_t2[0] - x2, pt2_t2[1] - y2],
                                [pt3_t2[0] - x2, pt3_t2[1] - y2]], dtype=np.int32)
            cv2.fillConvexPoly(cropped_triangle2_mask, points2, (255, 0, 0))
            # cropped_triangle_1 = cv2.bitwise_and(cropped_triangle1, cropped_triangle1, mask=cropped_triangle1_mask)

            points2 = np.float32(points2)

            M1 = cv2.getAffineTransform(points1, points2)

            warped_triangle1 = cv2.warpAffine(cropped_triangle1, M1, (w1, h1))

            M2 = cv2.getAffineTransform(points2, points1)

            warped_triangle2 = cv2.warpAffine(cropped_triangle2, M2, (w2, h2))

            cv2.imshow("t1 -> t2", warped_triangle1)
            cv2.imshow("t1 -> t2", warped_triangle1)

            #new_face_2[y: y + h, x: x + w] = warped_triangle1

            break

            #triangle1 = np.array([pt1_t1, pt2_t1, pt3_t1], dtype=np.int32)
            #rect1 = cv2.boundingRect(triangle1)
            #(x, y, w, h) = rect1
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #break

        #for t in triangles2:
        #    pt1 = (t[0], t[1])
        #    pt2 = (t[2], t[3])
        #    pt3 = (t[4], t[5])

        #    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        #    cv2.line(frame, pt2, pt3, (0, 0, 255), 2)
        #    cv2.line(frame, pt1, pt3, (0, 0, 255), 2)

        #    break

        #print(triangles_2)



        cv2.imshow("Face swap", frame)

    else:
        cv2.imshow("Face swap", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
