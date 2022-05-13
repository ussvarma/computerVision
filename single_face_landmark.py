import cv2
import mediapipe as mp

try:
    def draw_landmark(path):
        """
         this function draws a single face landmark on nose

        """
        # Initiate holistic model
        mp_face_mesh = mp.solutions.holistic
        holistic = mp_face_mesh.Holistic(min_detection_confidence=0.5)
        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            ret, image = cap.read()
            # image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
            if ret is not True:
                break
            height, width, _ = image.shape
            # Recolor Feed
            rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Make Detections
            result = holistic.process(rgb_img)
            # print(dir(result.face_landmarks))
            # print(result.face_landmarks.landmark)
            # for hand_landmarks in result.face_landmarks.landmark:
            for i in range(20):
                pt1 = result.face_landmarks.landmark[1]
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

path = "datasets/video3.mp4"

draw_landmark(path)
