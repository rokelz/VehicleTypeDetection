"""
@author: R.K.Opatha
"""

import numpy as np
import os.path
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
from classifier import FrameExtractor
import matplotlib.patches as patches
from skimage import io
import time
import argparse
from PIL import Image
import re
import time
import glob
import scipy.ndimage
import xml.etree.cElementTree as ET
import tensorflow as tf
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sort import Sort
from detector import GroundTruthDetections
#from haar import HaarDetect




def imagePrediction(dataImage):
    
    image_data = dataImage

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            break
     
    return human_string,score

def Extract(filepathIn):

    vidcap = cv2.VideoCapture(filepathIn)

    success,image = vidcap.read()
    count = 1
    success = True
    while success:
	success,image = vidcap.read()
	#print('Read a new frame: ', success)
	cv2.imwrite('test/Pictures%d.jpg' % count, image)     # save frame as JPEG file
	count += 1
    Classsify('start')	

def Classsify(start):
    pastx = 0
    pasty = 0
    pastw = 0
    pasth = 0		
    face_cascade = cv2.CascadeClassifier('cascade.xml')
    path = 'test/'
    listing = os.listdir(path)
    listing.sort(key=lambda f: int(filter(str.isdigit, f)))
     #colorArr={[255,204,204],[255,229,204],[255,255,204],[229,255,204],[204,255,204],[204,255,229],
	#[204,255,255],[204,229,255],[204,204,255],[229,204,255]}
    frameIndex = []
		#for num in range(0,10):
		#	frameIndex[num] = []

		#pastDetArr={}
    count1 = 0
    count2 = 0
	#polygon = Polygon([(2,243), (233,220), (620,221), (698,386)])
    polygon = Polygon([(690,411),(1512,477),(1537,1039),(307,985),(307,556)])

    count = 1
    ncars = 0
    frameCount = 0
    tifCounter = len(glob.glob1('test/',"*.jpg"))
    for infile in listing:
		   # print "Working on Detection"
	if frameCount==tifCounter-1:
	    	    break
	if infile.endswith(".jpg"):
	    #print "current file is: " + os.path.splitext(infile)[0];
	    frame = cv2.imread('test/'+infile)
	    cars = face_cascade.detectMultiScale(frame, 1.1, 2)      
	    frameCount += 1
	    for (x,y,w,h) in cars:
		    centerx = (x+x+w)/2
		    centery = (y+y+h)/2
		    point = Point(centerx, centery)
		    if(polygon.contains(point)):
			ncars = ncars + 1
			cv2.circle(frame,(centerx,centery),40,(0,255,255),1)
			file = open("cctv.txt", "a")
			if (frameCount % 4 == 1):
				
				dataTobeWritten = "0,"+str(frameCount-1)+",0,1,0,0,0,0," +str(x)+","+str(y)+","+str(x+w)+","+str(y+h) + "\n"
				pastx = x
				pasty = y
				pastw = w
				pasth = h
				
			else:
				dataTobeWritten = "0,"+str(frameCount-1)+",0,1,0,0,0,0," +str(pastx)+","+str(pasty)+","+str(pastx+pastw)+","+str(pasty+pasth) + "\n"
			file.write('{}'.format(dataTobeWritten))
			file.close()
	    if frameCount==tifCounter:
	    	    break
			   #cv2.imshow("Result",frame)
			   #im = Image.fromarray(frame)
			   #im.save("image_"+str(frameCount)+"_countmage_"+str(ncars)+".jpeg")
	    ncars = 0
	    cv2.waitKey(1)

def getVideo():
    cmd = './vid.sh'
    os.system(cmd)

def main():
    args = parse_args()
    
    display = args.display
    use_dlibTracker  = args.use_dlibTracker
    saver = args.saver
    Extract(args.pathIn)
    #fextract.Classsify('start')
    total_time = 0.0
    total_frames = 0

    globalVehicleCount = 0
    globalBikeCount = 0
    globalBusCount = 0
    globalCarCount = 0
    globalVanCount = 0
    globalLorryCount = 0
    globalTrishoCount = 0
    types = ''        
    generatedVehicleId = set()

    #tansorflow environmental variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #filename =sys.argv[1]
    #img = cv2.imread(sys.argv[1])
    #img1 = Image.open(sys.argv[1])
    #resized = img[65:200, 250:631]/255 # Pre-processing the image and normalize
    #wind_row, wind_col = 45,45 # dimensions of the image
    #img_rows, img_cols = 45,45

       # change this as you see fit
    image_path = 'imageout.jpg'
    #scale=7
    #y_len,x_len,_=img.shape



    # for disp
    if display:
        colours = np.random.rand(32, 3)  # used only for display
        plt.ion()
        fig = plt.figure()
    else:
        colours = np.random.rand(32, 3)  # used only for display
	#matplotlib.use('Agg')
        plt.ion()
        fig = plt.figure()


    if not os.path.exists('output'):
        os.makedirs('output')
    out_file = 'output/townCentreOut.top'

	#init haar detection results
    #haar_detection_results = HaarDetect()
    #haar_detection_results.haar()
	
    #init detector
    detector = GroundTruthDetections()

    #init tracker
    tracker =  Sort(use_dlib= use_dlibTracker) #create instance of the SORT tracker

    if use_dlibTracker:
        print "Dlib Correlation tracker activated!"
    else:
        print "Kalman tracker activated!"

    with open(out_file, 'w') as f_out:

        frames = detector.get_total_frames()
	frames = frames - 1
        #print(frames)
        
        for frame in range(0, frames):  #frame numbers begin at 0!
            # get detections
            detections = detector.get_detected_items(frame)
	    #print detections
            total_frames +=1
            fn = 'test/Pictures%d.jpg' % (frame + 1)  # video frames are extracted to 'test/Pictures%d.jpg' with ffmpeg
            img = io.imread(fn)
	    img1 = Image.open(fn)
            if (display):
                ax1 = fig.add_subplot(111, aspect='equal')
                ax1.imshow(img)
                if(use_dlibTracker):
                    plt.title('Dlib Correlation Tracker')
                else:
                    plt.title('Kalman Tracker')
	    else:
                ax1 = fig.add_subplot(111, aspect='equal')
                ax1.imshow(img)
            start_time = time.time()
            #update tracker
            trackers = tracker.update(detections,img)

            cycle_time = time.time() - start_time
            total_time += cycle_time

            #print('frame: %d...took: %3fs'%(frame,cycle_time))
            
            for d in trackers:
                f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[4], frame, 1, 1, d[0], d[1], d[2], d[3]))
		#print '%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[4], frame, 1, 1, d[0], d[1], d[2], d[3])
		img2 = img1.crop((d[0],d[1],d[2],d[3]))
	    	img2.save("imageout.jpg")
	    	image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            	classof,prediction = imagePrediction(image_data) # predict the image	
		
				
		#print (generatedVehicleId)
		if(d[4] not in generatedVehicleId):
			#print 'added : %d' % (d[4])
			generatedVehicleId.add(d[4])
			if(classof!="negative images"):
				globalVehicleCount += 1

				if(classof =="bike"):
					globalBikeCount += 1
					types = 'bike'
				if(classof =="bus"):
					globalBusCount += 1
					types = 'bus'
				if(classof =="car"):
					globalCarCount += 1
					types = 'car'
				if(classof =="van"):
					globalVanCount += 1
					types = 'van'
				if(classof =="lorry"):
					globalLorryCount += 1
					types = 'lorry'
				if(classof =="trisho"):
					globalTrishoCount += 1
					types = 'trisho'
		#else:
			#print 'have : %d' % (d[4])
			#print 'type : %s' % types
		

                if (display):
                    d = d.astype(np.int32)
                    ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                    ec=colours[d[4] % 32, :]))
                    ax1.set_adjustable('box-forced')
                    #label
                    ax1.annotate('%s(%d)' % (types,d[4]), xy=(d[0], d[1]), xytext=(d[0], d[1]))
                    if detections != []:#detector is active in this frame
                        ax1.annotate(" DETECTOR", xy=(5, 45), xytext=(5, 45))
		else:
 		    d = d.astype(np.int32)
                    ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                    ec=colours[d[4] % 32, :]))
                    ax1.set_adjustable('box-forced')
                    #label
                    ax1.annotate('%s(%d)' % (types,d[4]), xy=(d[0], d[1]), xytext=(d[0], d[1]))
                    if detections != []:#detector is active in this frame
                        ax1.annotate(" DETECTOR", xy=(5, 45), xytext=(5, 45))

            if (display):
                plt.axis('off')
                #fig.canvas.flush_events()
                plt.draw()
                fig.tight_layout()
                #save the frame with tracking boxes
                if(saver):
                    fig.savefig("frameOut/f%d.jpg"%(frame+1),dpi = 200)
                ax1.cla()
	    else:
		plt.axis('off')
                #fig.canvas.flush_events()
                plt.draw()
                fig.tight_layout()
		fig.savefig("frameOut/f%d.jpg"%(frame+1),dpi = 200)
		ax1.cla()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
    print("Tracked Vehicle Statistics \nBikes \t\t: %d \nBuses \t\t: %d \nCarss \t\t: %d \nVans \t\t: %d \nLorries \t: %d \nTrishos \t: %d \nTotal Vehicles \t: %d "%(globalBikeCount,globalBusCount,globalCarCount,globalVanCount,globalLorryCount,globalTrishoCount,globalVehicleCount))
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Experimenting Trackers with SORT')
    parser.add_argument('--NoDisplay', dest='display', help='Disables online display of tracker output (slow)',action='store_false')
    parser.add_argument('--dlib', dest='use_dlibTracker', help='Use dlib correlation tracker instead of kalman tracker',action='store_true')
    parser.add_argument('--save', dest='saver', help='Saves frames with tracking output, not used if --NoDisplay',action='store_true')
    parser.add_argument("--pathIn", help="path to video")
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    main()
