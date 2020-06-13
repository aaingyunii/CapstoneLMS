from django.shortcuts import render, redirect
from django.views.generic.list import ListView
from django.views.generic.edit import UpdateView, CreateView, DeleteView
from django.views.generic.detail import DetailView
from .models import Photo,Eve
from django.http import HttpResponseRedirect
from django.contrib import messages
from django.contrib.auth.models import User


########## 눈인식을 위한 코드 from azurr
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer

###
import numpy as np
import cv2
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from scipy.spatial import distance as dist

####
import imutils
import time
import timeit
import matplotlib.pyplot as plt
from . import make_train_data as mtd
from . import light_remover as lr
from . import ringing_alarm as alarm
import face_recognition
import matplotlib.pyplot as plt
from sklearn import metrics
import pygame
import threading

from .models import Eve
from .models import Threshold
from datetime import datetime
from django.contrib import auth

def eye_aspect_ratio(eye) :
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def init_open_ear() :
    time.sleep(5)
    print("open init time sleep")
    ear_list1 = []
    th_message1 = Thread(target = init_message)
    th_message1.daemon = True
    th_message1.start()
    print("Hi1")
    for i in range(7) :
    	print("Hi2")
    	ear_list1.append(both_ear)
    	time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list1) / len(ear_list1)
    print("open list =", ear_list1, "\nOPEN_EAR =", OPEN_EAR, "\n")

def init_close_ear() : 
    time.sleep(2)
    th_open.join()
    time.sleep(5)
    print("close init time sleep")
    ear_list2 = []
    th_message2 = Thread(target = init_message)
    th_message2.daemon = True
    th_message2.start()
    time.sleep(1)
    for i in range(7) :
        ear_list2.append(both_ear)
        time.sleep(1)
    global CLOSE_EAR
    CLOSE_EAR = sum(ear_list2) / len(ear_list2)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) #EAR_THRESH means 50% of the being opened eyes state
    print("close list =", ear_list2, "\nCLOSE_EAR =", CLOSE_EAR, "\n")
    print("The last EAR_THRESH's value :",EAR_THRESH, "\n")


def init_message():
    print("init_message")
    print("졸음인식시작!")
    alarm.sound_alarm("C:\\Users\\InKunAhn\\Documents\\GitHub\\CapstoneLMS\\sound\\init_sound.mp3")


# 코드 잠수확인용
def printForTest():
	print("코드 진행중입니다.")


def testForThreshold():
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
    
    #cv2.VideoCapture(0)에서 웹캠이 하나면 0을 이고 2개이상일 경우, 0,1,2...으로 웹캠을 설정할 수 있다.
    video_capture = cv2.VideoCapture(0)   
    count =0
  
    while th_close.is_alive():

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


                    print("Present both eyes value : ",both_ear)

        cv2.imshow('Frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    Threshold_instance=Threshold.objects.create(open_ear=OPEN_EAR,close_ear=CLOSE_EAR,ear_thresh=EAR_THRESH, created_at=datetime.now())
    print("테스트완료")

    

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
    
    #cv2.VideoCapture(0)에서 웹캠이 하나면 0을 이고 2개이상일 경우, 0,1,2...으로 웹캠을 설정할 수 있다.
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


                    print("Present both eyes value : ",both_ear)

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
            Eve_instance = Eve.objects.create(number=i,result=result_data[i],state='power', created_at = datetime.now())
        
        elif result_data[i] == 1:
	        Eve_instance = Eve.objects.create(number=i,result=result_data[i],state = 'normal', created_at = datetime.now())

        else: 
	        Eve_instance = Eve.objects.create(number=i,result=result_data[i],state = 'short', created_at = datetime.now())   


#### 기존 코드

class PhotoList(ListView):
    model = Photo
    template_name_suffix = '_list'

class PhotoCreate(CreateView):
    model = Photo
    fields = ['text', 'image']
    template_name_suffix = '_create'
    success_url = '/'

    def form_valid(self, form):
        form.instance.author_id=self.request.user.id
        if form.is_valid():
            #올바르다면
            #form: 모델폼
            form.instance.save()
            return redirect('/')
        else:
            #올바르지 않다면
            return self.render_to_response({'form': form})

class PhotoUpdate(UpdateView):
    model = Photo
    fields = ['text', 'image']
    template_name_suffix = '_update'
    success_url = '/'

    def dispatch(self, request, *args, **kwargs):
        object = self.get_object()
        if object.author != request.user:
            messages.warning(request, '수정할 권한이 없습니다.')
            return HttpResponseRedirect('/')
            #삭제 페이지에서 권한이 없다고 띄우거나 detail페이지로 들어가서 삭제에 실패했다 라고 띄우거나
        else:
            return super(PhotoUpdate, self).dispatch(request, *args, **kwargs)

class PhotoDelete(DeleteView):
    model = Photo
    template_name_suffix = '_delete'
    success_url = '/'

    def dispatch(self, request, *args, **kwargs):
        object = self.get_object()
        if object.author != request.user:
            messages.warning(request, '삭제할 권한이 없습니다.')
            return HttpResponseRedirect('/')
            #삭제 페이지에서 권한이 없다고 띄우거나 detail페이지로 들어가서 삭제에 실패했다 라고 띄우거나
        else:
            return super(PhotoDelete, self).dispatch(request, *args, **kwargs)

class PhotoDetail(DetailView):
    model = Photo
    template_name_suffix = '_detail'


    #눈인식 전체 동작을 위한 것.
    def dispatch(self, request, *args, **kwargs):
        # object = self.get_object()
        if (request.GET.get('btn')):
            th_drow=Thread(target=drowsiness())
            print("눈인식 시작")
            th_drow.daemon=True
            th_drow.start()
            return HttpResponseRedirect('/')
        else:

            return super(PhotoDetail, self).dispatch(request, *args, **kwargs)

    
    # def eye(request):
    #     if(request.GET.get('btn')):
    #         dr=drowsiness()
    # if(GET.get('btn')):
    #     dr=drowsiness()
    # else:
    #     # return HttpResponseRedirect('/')


def webcam_test(request):

    if(request.GET.get('eye_test')):
        th_test=Thread(target=testForThreshold)
        print("Test 시작")
        th_test.daemon=True
        th_test.start()

    return render(request, 'photo/webcam_test.html')

def mic_test(request):
    return render(request,'photo/mic_test.html')

def home(request):
    return render(request, 'photo/home.html')

def index(request):
    return render(request, 'photo/index.html', {})

from django.utils.safestring import mark_safe
import json

def room(request, room_name):
    return render(request, 'photo/room.html', {
        'room_name_json': mark_safe(json.dumps(room_name))
    })



