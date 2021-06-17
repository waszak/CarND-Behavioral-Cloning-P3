import os
import os.path
import csv
import time
import random
import numpy as np
import cv2
from scipy.spatial import ConvexHull 

def benchmark(dataset):
    start_time = time.perf_counter()
    i = 0
    for sample in dataset:
        i+=1
            
    print("Time:", time.perf_counter() - start_time)
    print("Number of batches:", i)

#when we shift image left or right we should also change steering by small number
def random_shift(image, yi=0):
    if random.random() < 0.5:
        value = np.random.randint(-20,20)
        M = np.float32([[1, 0, value ], [0, 1, 0]])
        #add random noise
        yi -= (value/20)* random.random() * 0.1
    #elif random.random() < 0.5:
    #    M = np.float32([[1, 0, 0], [0, 1, np.random.randint(-20,20)]])
    else:
        return image, yi
        
    translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return translated, yi
    
def generate_shadow(image): 
    positions_x = []   
    positions_y = []
    shape = image.shape
    mask = np.zeros_like(image, dtype=np.int32)
    for i in range(np.random.randint(5,30)): 
        x = random.randint(0, shape[1])
        y = random.randint(shape[0]//2, shape[0] )
        positions_x.append(x)
        positions_y.append(y)
    
    #positions_x = sorted(positions_x)
    #positions_y = sorted(positions_y)
    points = list(zip(positions_x, positions_y))
    hull = ConvexHull(points)
    #print(points)
    
    points = np.array(points, dtype=np.int32)
    points = points[hull.vertices]
    #print(hull.vertices)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    cv2.fillConvexPoly(mask, np.int32([points]) , 255) 
    shadow_ratio = random.uniform(0.4, 0.5)  
    hls[:,:,1][mask[:, :, 0] == 255] = hls[:, : , 1][mask[:,:,0] == 255] *  shadow_ratio 
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) 
 
 
def get_data(path):
    num_rows = 0 
    images = []
    measurements = []
    for directory in os.listdir(path):
        csv_dir = path+'\\'+directory
        with open(csv_dir+'\\driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] =='center':
                    #skip header if exists
                    continue 
                
                paths= [ csv_dir+'\\IMG\\'+os.path.basename(x) for x in row[0:3]]
                #print(paths)
                images.extend(paths)
                shift = 0.25
                measurements.append(float(row[3]))
                measurements.append(float(row[3])+0.25)
                measurements.append(float(row[3])-0.25)
                num_rows +=3
                
    return num_rows, images, measurements

     