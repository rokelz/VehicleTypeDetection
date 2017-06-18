import os, sys
import cv2
import numpy as np

from PIL import Image
import re
import time
import scipy.ndimage
import xml.etree.cElementTree as ET
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
filename =sys.argv[1]
img = cv2.imread(sys.argv[1])
img1 = Image.open(sys.argv[1])
resized = img[65:200, 250:631]/255 # Pre-processing the image and normalize
wind_row, wind_col = 45,45 # dimensions of the image
img_rows, img_cols = 45,45

# change this as you see fit
image_path = 'imageout.jpg'
scale=7
y_len,x_len,_=img.shape
# Read in the image_data


def imagePrediction(dataImage):
    
    image_data = dataImage

    with tf.Session() as sess:
     # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        #simg= cv2.resize(image_data,dsize=(299,299), interpolation = cv2.INTER_CUBIC)
#Numpy array
        #np_image_data = tf.convert_to_tensor(image_data)
        #np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
#maybe insert float convertion here - see edit remark!
        #np_final = np.expand_dims(np_image_data,axis=0)
        predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        #print('%s' % os.path.splitext(filename)[0])
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
         #   print('%s (score = %.5f)' % (human_string, score))
            break
    
 
    return human_string,score

def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def deeplearn():
    height, width, channels = scipy.ndimage.imread(filename).shape
    imageidentifier= filename.split("_")[0]
    root = ET.Element("annotations")
    folder = ET.SubElement(root, "folder").text=imageidentifier
    filenames = ET.SubElement(root, "filename").text=os.path.splitext(filename)[0]
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "ImageNet Database"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text =  str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(3)
    segmented = ET.SubElement(root, "segmented").text=str(0)
    objects = ET.SubElement(root, "object")
    ET.SubElement(objects, "name").text =  imageidentifier
    ET.SubElement(objects, "pose").text =  "Unspecified"
    ET.SubElement(objects, "difficult").text = str(0)

    for (x,y,w,h) in cars:
        	
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)      
            img2 = img1.crop((x,y,x+w,y+h))
	    img2.save("imageout.jpg")
	    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            classof,prediction = imagePrediction(image_data) # predict the image
            #classes = prediction[0]
	    

	    
	    
	    if(classof!="negative images"):	
                print("%s detected with a probability: %.5f"%(classof,prediction))
	        objects = ET.SubElement(root, "object")
    	        ET.SubElement(objects, "name").text =  classof
    	        ET.SubElement(objects, "pose").text =  "Unspecified"
    	        ET.SubElement(objects, "difficult").text = str(0)
                bndbox = ET.SubElement(objects, "bndbox")
                ET.SubElement(bndbox, "xmin").text =str(x)
                ET.SubElement(bndbox, "ymin").text =str(y)
                ET.SubElement(bndbox, "xmax").text =str(x+w)
                ET.SubElement(bndbox, "ymax").text =str(y+h)
            
	    else:
	        print("negative image")
	    cv2.waitKey(1)
            time.sleep(0.25)

    tree = ET.ElementTree(root)
    xmlfilename = str(os.path.splitext(filename)[0])
    tree.write(xmlfilename+".xml")
            #cv2.imshow("sliding_window", img[y:y+x_len,x:x+y_len])
            #cv2.imshow("Window", clone)
            
            #time.sleep(0.25)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
               in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

cascade_src = 'cascade.xml'
#video_src = 'image_6104.jpg'
#video_src = 'dataset/video2.avi'

#img = cv2.imread(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)



colorconversion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.line(img,(0,420),(640,420),(0,0,255),2)
cv2.line(img,(0,210),(640,210),(0,0,255),2)
  
cars = car_cascade.detectMultiScale(colorconversion, 1.1, 1)


deeplearn()

