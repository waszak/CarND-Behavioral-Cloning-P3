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
#0.00416 is constant picked by me.
#Left/Right camera have about 60px shift compared to center image.
#I use 0.25 as stering correction for left/right image.
#Constant is derived from diving 0.25 by 60
#I don't take into account top/bottom shifts. Its more to simulate move up/down hill.
def random_shift(image, yi=0):
    if random.random() < 0.7:
        tx = np.random.randint(-60,60)
        ty = np.random.randint(-10,10)
        M = np.float32([[1, 0, tx ], [0, 1, ty]])
        #add random noise
        #60px = 0.25
        yi += 0.00416 * tx
    #elif random.random() < 0.5:
    #    M = np.float32([[1, 0, 0], [0, 1, ]])
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
    shadow_ratio = random.uniform(0.4, 0.6)  
    hls[:,:,1][mask[:, :, 0] == 255] = hls[:, : , 1][mask[:,:,0] == 255] *  shadow_ratio 
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) 
 
def get_data(path):
    num_rows = 0 
    images = []
    measurements = []
    for directory in os.listdir(path):
        csv_dir = os.path.join( path, directory)
        csv_path = os.path.join(csv_dir, 'driving_log.csv')
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] =='center':
                    #skip header if exists
                    continue 
               
        
                paths= [ os.path.join( os.path.join(csv_dir,'IMG'), os.path.basename(x.replace('\\',os.sep))) for x in row[0:3]]
                images.extend(paths)
                shift = 0.25
                ratio = float(row[3])                      
                measurements.append(ratio)
                measurements.append(ratio + shift)
                measurements.append(ratio - shift)
                num_rows += 3
             
                
    return num_rows, images, measurements