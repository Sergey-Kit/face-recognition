import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm

#Folder
proj_path = os.path.abspath('')
print(proj_path)
proj_path

Data_folder = "VGGFace2_Data"
proj_path
file_names = os.listdir(os.path.join(proj_path, Data_folder, "./train/n000012/"))
#print(file_names)

#Detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(proj_path, './shape_predictor_68_face_landmarks.dat'))
#cap = cv2.VideoCapture(0)

# Read imgs
for filename in tqdm(file_names): 
#    #image_2 = cv2.imread("/mnt/c/Python Projects/Seal_Project/Seal_Data/raw_data_25/" + filename)
#    image_1 = cv2.imread(os.path.join(proj_path, Data_folder, "./raw_data_25/") + filename)
    image_1 = cv2.imread(os.path.join(proj_path, Data_folder, "./train/n000012/") + filename)
    gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
	# Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
	    # Draw the circle to mark the keypoint 
            cv2.circle(image_1, (x, y), 1, (0, 0, 255), -1)
		
    # Display the image
    cv2.imshow('Landmark Detection', image_1)
    cv2.imwrite(os.path.join(proj_path, "./Point_detector/") + filename, image_1)
    ## Press the escape button to terminate the code
    #if cv2.waitKey(10) == 27:
    #    break
