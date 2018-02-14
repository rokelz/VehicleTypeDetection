
import os, sys
import cv2
import numpy as np
import click
from PIL import Image
import re
import math
import time
import scipy.ndimage
import numpy as np
import os.path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import time
import argparse
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

 




class FrameExtractor:

    def Extract(self,filepathIn):

		vidcap = cv2.VideoCapture(filepathIn)

		success,image = vidcap.read()

		count = 1

		success = True

		while success:

			success,image = vidcap.read()

			#print('Read a new frame: ', success)

			cv2.imwrite('test/Pictures%d.jpg' % count, image)     # save frame as JPEG file

			count += 1

		self.Classsify('start')	

    def Classsify(self,start):
		
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

		    if infile.endswith(".jpg"):
			   print "current file is: " + os.path.splitext(infile)[0];
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
					dataTobeWritten = "0,"+str(frameCount-1)+",0,1,0,0,0,0," +str(x)+","+str(y)+","+str(x+w)+","+str(y+h) + "\n"
					file.write('{}'.format(dataTobeWritten))
					file.close()
			   if frameCount==tifCounter:
			   	break
			   #cv2.imshow("Result",frame)
			   #im = Image.fromarray(frame)
			   #im.save("image_"+str(frameCount)+"_countmage_"+str(ncars)+".jpeg")
			   ncars = 0
		   	   cv2.waitKey(1)


		


