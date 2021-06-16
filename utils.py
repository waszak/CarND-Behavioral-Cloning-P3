import os
import os.path
import csv

    
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
                
                paths= [ csv_dir+'\\IMG\\'+os.path.basename( os.path.normpath (x )) for x in row[0:3]]
                #print(paths)
                images.extend(paths)
                shift = 0.25
                measurements.append(float(row[3]))
                measurements.append(float(row[3])+0.25)
                measurements.append(float(row[3])-0.25)
                num_rows +=3
                
    return num_rows, images, measurements

     