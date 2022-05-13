import cv2
import mediapipe as mp

try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.2)
    cap = cv2.VideoCapture("datasets/video.mp4")
    # Initiate holistic model
    while cap.isOpened():
        ret, image = cap.read()
        # image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
        if ret is not True:
            break
        height, width, _ = image.shape
        # Recolor Feed
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make Detections
        result = face_mesh.process(rgb_img)

        #print(result.multi_face_landmarks)
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(20):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                # print(x,y)
                cv2.circle(image, (x, y), 7, (100, 0, 100), 5)
            cv2.imshow("img", image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(e)