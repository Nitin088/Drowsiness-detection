import cv2
from scipy.spatial import distance
import dlib


def EAR_value_single(eye):
#calculate the euclidean dist of eyes both vertical and horizontal
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
#calculate eye aspect ratio
	EAR = (A+B)/(2.0*C)
	return EAR
#initiliaze the hog face detector and the video cpature
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(r"D:\Nitin\pythonProject\shape_predictor_68_face_landmarks.dat")
#threshold ear value , if ear is lesser than threshold then eyes are closed
threshold_EAR=0.26
#loop to anlyze the frames of video
while True:
    _, frame = cap.read()
	#convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#detect faces in frames
    faces = hog_face_detector(gray,0)
    for face in faces:
        dlib_face_landmarks = dlib_facelandmark(gray, face)
        left_eye = []
        right_eye = []
		#detect left eye landmarks in dlib face detector
        for n in range(36,42):
        	x = dlib_face_landmarks.part(n).x
        	y = dlib_face_landmarks.part(n).y
        	left_eye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = dlib_face_landmarks.part(next_point).x
        	y2 = dlib_face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        #detect right eye landmarks in dlib face detector
        for n in range(42,48):
        	x = dlib_face_landmarks.part(n).x
        	y = dlib_face_landmarks.part(n).y
        	right_eye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = dlib_face_landmarks.part(next_point).x
        	y2 = dlib_face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        #calculate the eye aspect ratio for each eye
        left_ear = EAR_value_single(left_eye)
        right_ear = EAR_value_single(right_eye)
        #calculate the average EAR value
        final_EAR = (left_ear+right_ear)/2
        final_EAR = round(final_EAR,2)


		#if EAR falls below the threshold value display message in video
        if final_EAR<threshold_EAR:
        	cv2.putText(frame,"DROWSY",(20,100),
        		cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
        	cv2.putText(frame,"Are you Sleepy?",(20,400),
        		cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
        	print("Drowsy")
        print(final_EAR)

    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == ord("e"):
        break
cap.release()
cv2.destroyAllWindows()