import cv2
import dlib
import numpy as np
import os
#from scipy.spatial import procrustes
from tqdm import tqdm

proj_path = os.path.abspath('')
print(proj_path)
proj_path

Data_folder = "VGGFace2_Data"
proj_path
file_names = os.listdir(os.path.join(proj_path, Data_folder, "./train_mini/n000012/"))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(proj_path, './shape_predictor_68_face_landmarks.dat'))

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}
   
    return d, Z, tform



def make_keypoints(file_names, detector, predictor):

    # Load the reference shape from the first image
    ref_filename = file_names[0]
    ref_image = cv2.imread(os.path.join(proj_path, Data_folder, "./train_mini/n000002/") + ref_filename)
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_rects = detector(ref_gray, 1)
    ref_shape = None

    # Get the landmark points for the reference image
    for ref_rect in ref_rects:
        ref_shape = predictor(ref_gray, ref_rect)
        ref_shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            ref_shape_np[i] = (ref_shape.part(i).x, ref_shape.part(i).y)
        ref_shape = ref_shape_np

    for filename in tqdm(file_names[0:3]): 
        image_1 = cv2.imread(os.path.join(proj_path, Data_folder, "./train_mini/n000002/") + filename)
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

            # Apply Procrustes analysis
            _, _, tform = procrustes(shape, ref_shape)

            # Apply the transformation to the landmark points
            transformed_shape = (np.dot(shape, np.array(tform['rotation'])) + np.array(tform['translation'])).astype(int)
            
            # Display the landmarks
            image_2 = image_1.copy()
            for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint 
                image_pointed = cv2.circle(image_2, (x, y), 1, (0, 0, 255), -1)

            # Display the Procrustes landmarks
            image_3 = image_1.copy()
            for i, (x, y) in enumerate(transformed_shape):
            # Draw the circle to mark the keypoint 
                image_procr_pointed = cv2.circle(image_3, (x, y), 1, (0, 0, 255), -1)
        

        # Display & save the images
        cv2.imshow('Landmark Detection', image_pointed)
        cv2.imwrite(os.path.join(proj_path, "./Point_detector/") + filename, image_pointed)
        cv2.imwrite(os.path.join(proj_path, "./Procrustes_points/") + filename, image_procr_pointed)
        
        print(rects)
        

    
make_keypoints(file_names, detector, predictor)



