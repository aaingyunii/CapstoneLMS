from django.shortcuts import render
# from drowsiness.drowsiness_detector import *

# def drowsiness(request):

# 	return render(request,'index.html',{'detector':drowsiness_detector})

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
# from check_cam_fps.py import check_fps
###
import numpy as np
import cv2
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from scipy.spatial import distance as dist

####
import numpy as np
import imutils
import time
import timeit
import dlib
import matplotlib.pyplot as plt
# import make_train_data as mtd
# import light_remover as lr
# import ringing_alarm as alarm
import face_recognition
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pygame
import threading

from .models import Eve
from datetime import datetime

# class check_cam_fps:
# 	import time

#	현재 사용하지 않음으로 주석처리됨
# 	def check_fps(prev_time) :
# 	    cur_time = time.time() #Import the current time in seconds
# 	    one_loop_time = cur_time - prev_time
# 	    prev_time = cur_time
# 	    fps = 1/one_loop_time
# 	    return prev_time, fps

#추가로 변수 미리 선언
knn = cv2.ml.KNearest_create()

class mtd:
	#%matplotlib inline
	#plt.style.use('ggplot')

	#'num_samples' is number of data points to create
	#'num_features' means the number of features at each data point (in this case, x-y conrdination values)

	def start(sample_size=25) :

	    train_data = generate_data(sample_size)
	    #print("train_data :",train_data)
	    labels = classify_label(train_data)
	    power, normal, short = binding_label(train_data, labels)
	    print("Return true if training is successful :", knn.train(train_data, cv2.ml.ROW_SAMPLE, labels))
	    return power, normal, short

	def run(new_data, power, normal, short):
	    a = np.array([new_data])
	    b = a.astype(np.float32)
	    #plot_data(power, normal, short)    
	    ret, results, neighbor, dist = knn.findNearest(b, 5) # Second parameter means 'k'
	    #print("Neighbor's label : ", neighbor)
	    print("predicted label : ", results)
	    #print("distance to neighbor : ", dist)
	    #print("what is this : ", ret)
	    #plt.plot(b[0,0], b[0,1], 'm*', markersize=14);
	    return int(results[0][0])

class lr:
	import cv2

	def light_removing(frame) :
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
	    L = lab[:,:,0]
	    med_L = cv2.medianBlur(L,99) #median filter
	    invert_L = cv2.bitwise_not(med_L) #invert lightness
	    composed = cv2.addWeighted(gray, 0.75, invert_L, 0.25, 0)
	    return L, composed

class alarm:

	def sound_alarm(path) :
		pygame.mixer.init()
		pygame.mixer.music.load(path)
		pygame.mixer.music.play()


	def select_alarm(result) :
		if result == 0:
			alarm.sound_alarm("C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\sound\\power_alarm.wav")
		elif result ==1 :
			alarm.sound_alarm("C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\sound\\normal_alarm.wav")
		else :
			alarm.sound_alarm("C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\sound\\short_alarm.mp3")

### mtd 파일안에 선언되었던 함수들
def generate_data(num_samples, num_features = 2) :
    """randomly generates a number of data points"""    
    data_size = (num_samples, num_features)
    data = np.random.randint(0,40, size = data_size)
    return data.astype(np.float32)

#I determined the drowsiness-driving-risk-level based on the time which can prevent driving accident.
def classify_label(train_data):
    labels = []
    for data in train_data :
        if data[1] < data[0]-15 :
            labels.append(2)
        elif data[1] >= (data[0]/2 + 15) :
            labels.append(0)
        else :
            labels.append(1)
    return np.array(labels)

def binding_label(train_data, labels) :
    power = train_data[labels==0]
    normal = train_data[labels==1]
    short = train_data[labels==2]
    return power, normal, short

def plot_data(po, no, sh) :
    plt.figure(figsize = (10,6))
    plt.scatter(po[:,0], po[:,1], c = 'r', marker = 's', s = 50)
    plt.scatter(no[:,0], no[:,1], c = 'g', marker = '^', s = 50)
    plt.scatter(sh[:,0], sh[:,1], c = 'b', marker = 'o', s = 50)
    plt.xlabel('x is second for alarm term')
    plt.ylabel('y is 10s for time to close eyes')


#We don't use below two functions. 
def accuracy_score(acc_score, test_score) :
    """Function for Accuracy Calculation"""
    print("KNN Accuracy :",np.sum(acc_score == test_score) / len(acc_score))
    #A line below this comment is exactly same with above one.
    #print(metrics.accuracy_score(acc_score, test_score))
    
def precision_score(acc_score, test_score) :
    """Function for Precision Calculation"""
    true_two = np.sum((acc_score == 2) * (test_score == 2))
    false_two = np.sum((acc_score != 2) * (test_score == 2))
    precision_two = true_two / (true_two + false_two)
    print("Precision for the label '2' :", precision_two)
    
    true_one = np.sum((acc_score == 1) * (test_score == 1))
    false_one = np.sum((acc_score != 1) * (test_score == 1))
    precision_one = true_one / (true_one + false_one)
    print("Precision for the label '1' :", precision_one)
    
    true_zero = np.sum((acc_score == 0) * (test_score == 0))
    false_zero = np.sum((acc_score != 0) * (test_score == 0))
    precision_zero = true_zero / (true_zero + false_zero)
    print("Precision for the label '0' :", precision_zero)


### lr안에 선언되었던 함수
def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# ### alarm 안에 선언되었던 함수
# def sound_alarm(path) :
#     pygame.mixer.init()
#     pygame.mixer.music.load(path)
#     pygame.mixer.music.play()
    

########################	drowsiness dector 원본 안의 코드.
def init_open_ear() :
    time.sleep(5)
    print("open init time sleep")
    ear_list = []
    th_message1 = Thread(target = init_message)
    th_message1.daemon = True
    th_message1.start()
    print("Hi1")
    for i in range(7) :
    	print("Hi2")
    	ear_list.append(both_ear)
    	time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)
    print("open list =", ear_list, "\nOPEN_EAR =", OPEN_EAR, "\n")



def init_close_ear() : 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list = []
    th_message2 = Thread(target = init_message)
    th_message2.daemon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7) :
        ear_list.append(both_ear)
        time.sleep(1)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) #EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :",EAR_THRESH, "\n")


def init_message() :
    print("init_message")
    print("졸음인식시작!")
    alarm.sound_alarm("C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\sound\\init_sound.mp3")

# 코드 잠수확인용
def printForTest():
	print("코드 진행중입니다.")



# Create your views here.
def drowsiness():
    KEY = 'e58722bb0e8e4bd9bcd627343c3e421f'
    ENDPOINT = 'https://recogface.cognitiveservices.azure.com/'
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))
    img_name = 'C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\identification\\inkun.jpg'
    image_data = open(img_name,'rb')
    detected_faces = face_client.face.detect_with_stream(image=image_data, recognition_model="recognition_02",)

    if not detected_faces:
        raise Exception('No face detected from image {}'.format(img_name))
    # Display the detected face ID in the first single-face image.
    # Face IDs are used for comparison to faces (their IDs) detected in other images.

    print('Detected face ID from', img_name, ':')
    for face in detected_faces: print (face.face_id)
    person1Id=face.face_id
    print()

    # Save this ID for use in Find Similar
    first_image_face_ID = detected_faces[0].face_id    
    
    #22222
    print("이제 시작한다!")

	#1.
    OPEN_EAR = 0 #For init_open_ear()
    global EAR_THRESH  #Threashold value
    EAR_THRESH = 0

	#2.
	#It doesn't matter what you use instead of a consecutive frame to check out drowsiness state. (ex. timer)
    EAR_CONSEC_FRAMES = 20 
    COUNTER = 0 #Frames counter.

	#3.
    closed_eyes_time = [] #The time eyes were being offed.
    TIMER_FLAG = False #Flag to activate 'start_closing' variable, which measures the eyes closing time.
    ALARM_FLAG = False #Flag to check if alarm has ever been triggered.

	#4. 
    ALARM_COUNT = 0 #Number of times the total alarm rang.
    RUNNING_TIME = 0 #Variable to prevent alarm going off continuously.

	#5.    
    PREV_TERM = 0 #Variable to measure the time eyes were being opened until the alarm rang.

	#6. make trained data 
    np.random.seed(9)
    power, normal, short = mtd.start(25) #actually this three values aren't used now. (if you use this, you can do the plotting)
	#The array the actual test data is placed.
    test_data = []
	#The array the actual labeld data of test data is placed.
    result_data = []

    print("?")

    print("Hi4")

	#9.
    global th_open
    th_open = Thread(target = init_open_ear)
    print("Hi5")
    th_open.daemon = True
    th_open.start()
    global th_close
    th_close = Thread(target = init_close_ear)
    th_close.daemon = True
    th_close.start()
    video_capture = cv2.VideoCapture(0)   
    count =0

    while True:
        global frame
        ret, frame = video_capture.read()
        if th_open.is_alive():
            cv2.putText(frame, "Open your both eyes until the sound end", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if (th_close.is_alive())&(th_open.is_alive()==False):
            cv2.putText(frame, "Close your both eyes until the sound end", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)   
        #L, gray = lr.light_removing(frame)
        if(int(video_capture.get(1)) % 1 == 0):
            cv2.imwrite("C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\frame\\%d.jpg" % count, frame)
            # print('Saved frame%d.jpg' % count)
            image_data1 = open("C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\frame\\%d.jpg" % count,'rb')
            detected_faces2 = face_client.face.detect_with_stream(image=image_data1,return_face_landmarks=True, recognition_model="recognition_02", )
            count += 1	
            person2Id=0
            if len(detected_faces2)==0:
                cv2.putText(frame, "Don't leave from Mornitor", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)   
            
            else:
                
                for face in detected_faces2: 
                    person2Id=face.face_id
                    (x,y)=(face.face_landmarks.eye_left_bottom.x,face.face_landmarks.eye_left_bottom.y)
                    (x1,y1)=(face.face_landmarks.eye_left_inner.x,face.face_landmarks.eye_left_inner.y)
                    (x2,y2)=(face.face_landmarks.eye_left_outer.x,face.face_landmarks.eye_left_outer.y)
                    (x3,y3)=(face.face_landmarks.eye_left_top.x,face.face_landmarks.eye_left_top.y)
                    leftEye=dist.euclidean((x,y),(x3,y3))/dist.euclidean((x1,y1),(x2,y2))
                    (x4,y4)=(face.face_landmarks.eye_right_bottom.x,face.face_landmarks.eye_right_bottom.y)
                    (x5,y5)=(face.face_landmarks.eye_right_inner.x,face.face_landmarks.eye_right_inner.y)
                    (x6,y6)=(face.face_landmarks.eye_right_outer.x,face.face_landmarks.eye_right_outer.y)
                    (x7,y7)=(face.face_landmarks.eye_right_top.x,face.face_landmarks.eye_right_top.y)
                    rightEye=dist.euclidean((x4,y4),(x7,y7))/dist.euclidean((x5,y5),(x6,y6))
                    global both_ear
                    both_ear=(leftEye+rightEye)*500
                    leftEyeShape=np.array([[x,y],[x1,y1],[x2,y2],[x3,y3]],dtype=np.int32)
                    rightEyeShape=np.array([[x4,y4],[x5,y5],[x6,y6],[x7,y7]],dtype=np.int32)
                    leftEyeHull = cv2.convexHull(leftEyeShape)
                    rightEyeHull = cv2.convexHull(rightEyeShape)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
                    # print (face.face_id)

            
                    result1=face_client.face.verify_face_to_face(person1Id, person2Id, person_group_id=None, 
                                                large_person_group_id=None, custom_headers=None, raw=False)
                    # print(result1.is_identical,result1.confidence)

                
                    if(result1.is_identical==False):
                        cv2.putText(frame, "Who are you?", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            
                    if both_ear < EAR_THRESH :

                        if not TIMER_FLAG:
                            start_closing=timeit.default_timer()
                            TIMER_FLAG=True
                        COUNTER+=1

                        if COUNTER >= EAR_CONSEC_FRAMES:
                            mid_closing = timeit.default_timer()
                            closing_time=round((mid_closing - start_closing),3)

                            if closing_time >= RUNNING_TIME:
                                if RUNNING_TIME ==0:
                                    CUR_TERM = timeit.default_timer()
                                    OPENED_EYES_TIME = round((CUR_TERM - PREV_TERM),3)
                                    PREV_TERM = CUR_TERM
                                    RUNNING_TIME = 1.75

                                RUNNING_TIME +=2
                                ALARM_FLAG = True
                                ALARM_COUNT +=1

                                print("{0}st Alarm".format(ALARM_COUNT))
                                print("The time eyes is being opened before the alarm went off :", OPENED_EYES_TIME)
                                print("closing time : ",closing_time)
                                test_data.append([OPENED_EYES_TIME, round(closing_time * 10,3)])
                                result = mtd.run([OPENED_EYES_TIME, closing_time * 10],power,normal,short)
                                result_data.append(result)
                                print("{0} drowsiness".format(result))
                                t = Thread(target = alarm.select_alarm, args=(result, ))
                                t.daemon=True
                                t.start()

                    else :
                        COUNTER = 0
                        TIMER_FLAG = False
                        RUNNING_TIME = 0

                        if ALARM_FLAG :
                            end_closing = timeit.default_timer()
                            closed_eyes_time.append(round((end_closing - start_closing),3))
                            print("The time eyes were being offed : ", closed_eyes_time)

                        ALARM_FLAG = False
            
                    cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (300,130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
    for i in range(len(result_data)):
    
        if result_data[i] == 0:
            Eve_instance = Eve.objects.create(name=i,result=result_data[i],state='power', created_at = datetime.now())
        
        elif result_data[i] == 1:
	        Eve_instance = Eve.objects.create(name=i,result=result_data[i],state = 'normal', created_at = datetime.now())

        else: 
	        Eve_instance = Eve.objects.create(name=i,result=result_data[i],state = 'short', created_at = datetime.now())   

	# return render(request,'index.html')

def main(request):
	print("good")

	if(request.GET.get('btn')):
		dr=drowsiness()

	return render(request,'index.html')

def showVideo(request):
    return render(request, 'videos.html')