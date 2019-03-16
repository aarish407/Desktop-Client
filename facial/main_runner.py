# imports
import sys
import cv2
# My libs
from face_detector import detector

# call the model(image) -> LIST OF BOUNDING BOXES as to where the faces could be
model = detector()      # object of class
list_faces = []         # List to store what we found in an image


# get image
img = cv2.imread('demo_pic.jpg')        # Later change it to a function and accept from there

# get positions
faces = model.get_positions(img)


'''
# For visual  analysis
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces Detected", img)
cv2.waitKey(0)
'''

# for each face you found
for face in faces:
    (x,y,w,h) = face
    
    # Crop face
    crop_img = img[y:y+h, x:x+w]
    
    # Store in list 
    list_faces.append(crop_img)
    
'''
# For visual analysis
cv2.imshow("cropped", list_faces[0])
cv2.waitKey(0)
'''

    # Align images
    # send to 'parameter calculation algorithm'
    # save parameters to dataset(pandas dataframe)
    # (Later Stage) Calculate Clusters and all