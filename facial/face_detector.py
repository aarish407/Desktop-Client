# algo to detect face positions and give out bounding boxes positions
# For now take haar cascade for quick prototyping
# Save the bounding boxes

# Haar Cascade 

import cv2
import sys

# Location for dependencies
path_img = "demo_pic.jpg"#sys.argv[1]
path_frontfacecascade = "haarcascade_frontalface_default.xml"

# Model 
model = cv2.CascadeClassifier(path_frontfacecascade)

# Import image 
img = cv2.imread(path_img)
# Gray conversion for haar requirement
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  


# faces will store the array of boxes of positions
faces = model.detectMultiScale(
    img_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(1,1),
    flags = cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces Detected", img)
cv2.waitKey(0)

