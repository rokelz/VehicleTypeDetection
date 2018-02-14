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

path = 'frameOut/'
listing = os.listdir(path)
listing.sort(key=lambda f: int(filter(str.isdigit, f)))
frameCount = 0
tifCounter = len(glob.glob1('frameOut/',"*.jpg"))
for infile in listing:
	frame = cv2.imread('frameOut/'+infile)
	if frameCount==tifCounter-1:
		break
	if infile.endswith(".jpg"):
		cv2.imshow("Result",frame)

	cv2.waitKey(1)
