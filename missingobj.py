# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import re
 
def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return (dx*dy)/((a[2]-a[0])*(a[3]-a[1]))
    else:
        return 0

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image



parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--image', default='present.jpg')
#parser.add_argument('--video', default='absent.mp4')
args = parser.parse_args()
check=str('absent.jpg')    
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights )
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
#    for i in net.getUnconnectedOutLayers():
#        print(layersNames[i[0] - 1])
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    global label1
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
#    print(conf)
    label = '%.2f' % conf    
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
        label1 = str(label)
#        print(label)
#        x=re.findall(r"^\w+",label)
        
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
objdim=list()
# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    global pc, oc, obs, count, objdim
    pc=0
    oc=0 
    objdim=list()
    obs = list()
    count = list()
    so=list()
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        so.append(classIds[i])
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

        x=re.findall(r"^\w+",label1)
#        print(x)
            #print(x[0])
        if x[0] not in obs:
            count.append(0)
            obs.append(x[0])    
        count[obs.index(x[0])]=count[obs.index(x[0])]+1
        if str(x[0]) == "person":
            objdim.append([0])
#               print('yay!!')
            pc = pc + 1
        else:
            objdim.append([left, top, left + width, top + height])
            oc = oc + 1
    objdim=[x for _,x in sorted(zip(so,objdim))]
    test=list()
    for i in obs:
        test.append(classes.index(i))
    obs=[x for _,x in sorted(zip(test,obs))]
    count=[x for _,x in sorted(zip(test,count))]
# Process inputs
winName = 'present'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
global before
before = list()

outputFile = "bottlethere.avi"
# Open the image file
if not os.path.isfile(args.image):
    print("Input image file ", args.image, " doesn't exist")
    sys.exit(1)
cap = cv.VideoCapture(args.image)
outputFile = 'bottlethere.jpg'

    
#while cv.waitKey(1) < 0:
    
    # get frame from the video
hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
if not hasFrame:
    print("Done processing !!!")
    print("Output file is stored as ", outputFile)
    cv.waitKey(3000)
    # Release device
    cap.release()

    # Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
net.setInput(blob)

    # Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
postprocess(frame, outs)
bod=objdim
#bod - dimension of objects in first image
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes

cv.imwrite(outputFile, frame.astype(np.uint8))
before=obs
bcount=count
print(before)
print(bcount)

print(pc,' faces found')
print(oc,' objects found');
cv.imshow(winName, frame)


#######################################
winName = 'bottlenotthere'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "notthere.avi"
# Open the image file
if not os.path.isfile(check):
    print("Input image file ", check, " doesn't exist")
    sys.exit(1)
cap = cv.VideoCapture(check)
outputFile = 'notthere.jpg'

    
#while cv.waitKey(1) < 0:
    
    # get frame from the video
hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
if not hasFrame:
    print("Done processing !!!")
    print("Output file is stored as ", outputFile)
    cv.waitKey(3000)
    # Release device
    cap.release()

    # Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
net.setInput(blob)

    # Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes

#cv.imwrite(outputFile, frame.astype(np.uint8))

print(obs)
print(count)
print(pc,' faces found')
print(oc,' objects found');


################################################################
#if not (set(before).issubset(set(obs))) and count != bcount::
if 'person' in before:
    cnt=bcount[before.index('person')]
else:
    cnt=0
    
if 'person' in obs:
    cnt1=count[obs.index('person')]
else:
    cnt1=0

for i in before :
    if(i!='person'):
        if i in obs: 
            if(count[obs.index(i)]<bcount[before.index(i)]):
    #            for j in range(bcount[before.index(i)]):                    
                one=bod[cnt:cnt+bcount[before.index(i)]]
                two=objdim[cnt1:cnt1+count[obs.index(i)]]
                '''
                miss=0
                for j in one:
                    if j not in two:
                        miss=miss+1
                        o=j
                        cv.rectangle(frame,(o[0],o[1]),(o[2],o[3]),(0,0,255),5)                    
                print(miss,' ',i,' missing')
                '''
                chec1=list()
                chec2=list()
                ct=bcount[before.index(i)]-count[obs.index(i)]
                for j in one: 
                    mi=10
                    mili=0
                    for k in two:
                        if area(j,k) < mi:
                            mili=j
                            mi=area(j,k)
                    chec1.append(mi)
                    chec2.append(mili)
                chec2=[x for _,x in sorted(zip(chec1,chec2))]
                dr=chec2[0:ct]
                for o in dr:
                    cv.rectangle(frame,(o[0],o[1]),(o[2],o[3]),(0,0,255),5)
                #print(bod[before.index(i)])
                cnt=cnt+bcount[before.index(i)]
                cnt1=cnt1+count[obs.index(i)]
            else:
                cnt=cnt+bcount[before.index(i)]
                cnt1=cnt1+count[obs.index(i)]
        else:
            for j in range(bcount[before.index(i)]):
                print('object ',cnt ,' ',i,' is missing')
                o=bod[cnt+j]
                cv.rectangle(frame,(o[0],o[1]),(o[2],o[3]),(0,0,255),5)
            cnt=cnt+bcount[before.index(i)]
#    else:
#        cnt=cnt+bcount[before.index(i)]
        
#if flag == 1:
#    o=objdim[before.index(i)]
#    cv.rectangle(frame,(o[0],o[1]),(o[2],o[3]),(0,0,255),5)
cv.imshow(winName, frame)
cv.imwrite(outputFile, frame.astype(np.uint8))