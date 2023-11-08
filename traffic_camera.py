import cv2
import numpy as np
video_path = "cars.mp4"
cap = cv2.VideoCapture(video_path)
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (960,960),scale = 1/255)
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,frame = cap.read()
    if not ret:
        print("video ended or does not play")
        break
    else:
        box_count = 0
        (class_ids,scores,bboxes) = model.detect(frame)
        height,width,_ = frame.shape
        print(height,width)
        #cv2.imshow("gray",gray)
        print("id: ",class_ids)
        print("scores: ",scores)
        print("boxes: ",bboxes)
        img_ =""
        
        for (class_id,score,bbox) in zip(class_ids,scores,bboxes):
            (x,y,w,h) = bbox 
            print(x,y,w,h)
            if(class_id==2):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                box_count = box_count +1
                if(box_count>10):
                    img_ = "Traffic"
                else:
                    img_ = "No Traffic"
        frame = cv2.putText(frame,img_,(500,height-800),font,4,(0,255,0),5,cv2.LINE_AA)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):

             break
       
cap.release()
cv2.destroyAllWindows()

