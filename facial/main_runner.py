# import the face_detector 
import sys
import cv2

from face_detector import detector

# call the model(image) -> LIST OF BOUNDING BOXES as to where the faces could be
# object of class
model = detector()

# get image
img = cv2.imread('demo_pic.jpg')

# get positions
faces = model.get_positions(img)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces Detected", img)
cv2.waitKey(0)

# for each face you found
    # crop face and align it
    # send to parameter calculation algorithm 
    # save parameters to dataset(pandas dataframe)
    # (Later Stage) Calculate Clusters and all