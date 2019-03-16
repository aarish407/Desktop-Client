# algo to detect face positions and bounding boxes positions
# For now take haar cascade for quick prototyping
# Save the bounding boxes

# Haar Cascade 

import cv2
import sys

path_img = sys.argv[0]
path_frontfacecascade = "haarcascade_frontalface_default.xml"



# for each face you found
    # crop face and align it
    # send to parameter calculation algorithm 
    # save parameters to dataset(pandas dataframe)
    # (Later Stage) Calculate Clusters and all