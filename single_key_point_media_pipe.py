import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
# creating facemesh instance
face_mesh = mp_face_mesh.FaceMesh()
#reading image
image = cv2.imread("datasets/srinidhi-shetty.jpg")
height, width, _ = image.shape
print(height, width)
#converting into rgb color
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = face_mesh.process(rgb_img)

# prints landmarks of face
print(result.multi_face_landmarks)

for facial_landmarks in result.multi_face_landmarks:
    pt1 = facial_landmarks.landmark[1]
    x = int(pt1.x * width)
    y = int(pt1.y * height)
    print(x, y)
    cv2.circle(image, (x, y), 7, (100, 0, 100), 5)
cv2.imshow("img", image)
cv2.waitKey(3000)
cv2.destroyAllWindows()
