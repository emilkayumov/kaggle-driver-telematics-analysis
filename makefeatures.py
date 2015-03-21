# -*- coding: utf-8 -*-

from os import listdir
import pandas
import numpy as np
from sklearn.externals import joblib    

DIR = '/home/emil/Code/Kaggle/driver telematics analysis/'
drivers = listdir(DIR + 'drivers')
countdriver = len(drivers)

result = np.empty((0,77))
drivernames = np.empty((countdriver))
countdone = 0

#for every driver
for driver in drivers:
    data = np.empty((200,77))
    drivernames[countdone] = driver

    #for every route
    for route in range(1,201):
        pwd = DIR + 'drivers/' + str(driver) + '/' + str(route) + '.csv'

        csvfile = pandas.read_csv(pwd)
        current = np.array(csvfile.values[:,:], dtype = "float64")

        #speed 
        size = current.shape[0]
        speed = np.sqrt(np.sum(np.power(current[1:,:]-current[:size-1,:], 2), axis=1))         
        
        #quantile of speed
        data[route-1,0:10] = np.percentile(speed, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            
        #acceleration
        acceleration = speed[1:] - speed[:speed.size-1]  

        #quantile of acceleration
        data[route-1,10:20] = np.percentile(acceleration, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])        
        
        #normal acceleration
        ax = current[:size-2,0]
        bx = current[1:size-1,0]
        cx = current[2:size,0]

        ay = current[:size-2,1]
        by = current[1:size-1,1]
        cy = current[2:size,1]

        d = 2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))      
        ux = ((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/d
        uy = ((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/d
                
        r = np.sqrt((ax-ux)*(ax-ux)+(ay-uy)*(ay-uy))
        normalacceleration = speed[1:size-1]*speed[1:size-1]/r
        normalacceleration[d == 0] = 0
        
        #quantile of acceleration
        data[route-1,20:30] = np.percentile(normalacceleration, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        #angle
        x_coor = (current[5:,0]-current[:size-5,0])
        y_coor = (current[5:,1]-current[:size-5,1])
        norm = np.sqrt(np.power(x_coor,2)+np.power(y_coor,2))
        x_coor /= norm
        y_coor /= norm
                
        cos_angle = np.abs(x_coor[1:]*x_coor[:x_coor.size-1]+y_coor[1:]*y_coor[:y_coor.size-1])
        cos_angle[np.isnan(cos_angle)] = 1
        cos_angle[cos_angle < 0.5] = 1
        
        count_angle = 0        
        for iter in range(1,cos_angle.size-2):
            if np.sum(cos_angle[iter-1:iter+2] < 0.99) == 3:
                count_angle += 1
        
        #quantile of angle cos        
        data[route-1,30:40] = np.percentile(cos_angle, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])    
        
        #speed param
        #mean speed without stop
        data[route-1,40] = np.sum(speed[speed > 0]) / np.sum(speed > 0)
        #mean speed with stop
        data[route-1,41] = speed.mean()
        #max speed        
        data[route-1,42] = speed.max()
        #speed deviation
        data[route-1,43] = speed.std()

        #acceleration param
        #mean acceleration without stop
        data[route-1,44] = np.sum(acceleration[acceleration != 0]) / np.sum(acceleration != 0)
        #mean acceleration with stop
        data[route-1,45] = acceleration.mean()
        #mean abs of acceleration with stop
        data[route-1,46] = np.abs(acceleration).mean()
        #max acceleration        
        data[route-1,47] = acceleration.max()
        #max deceleration
        data[route-1,48] = acceleration.min()
        #max acceleration or deceleration
        data[route-1,49] = np.abs(acceleration).max()        
        #accelertation deviation
        data[route-1,50] = acceleration.std()
        #mean acceleration and deceleration
        data[route-1,51] = acceleration[acceleration > 0].mean()
        data[route-1,52] = acceleration[acceleration < 0].mean()

        #normalacceleration param
        #mean without stop
        data[route-1,53] = (normalacceleration[normalacceleration != 0]).mean()
        #mean with stop
        data[route-1,54] = normalacceleration.mean()
        #max 
        data[route-1,55] = normalacceleration.max()
        #deviation 
        data[route-1,56] = normalacceleration.std()
        
        #angle param
        #mean all
        data[route-1,57] = cos_angle.mean()
        #mean without 1        
        data[route-1,58] = cos_angle[cos_angle != 1].mean()
        #min
        data[route-1,59] = cos_angle.min()
        #deviation
        data[route-1,60] = cos_angle.std()
        #count angle
        data[route-1,61] = count_angle / cos_angle.size
        #count angle per km
        data[route-1,62] = count_angle
                
        #distance
        data[route-1,63] = np.sum(speed)
        data[route-1,62] /= data[route-1,63]
        
        #moving
        data[route-1,64] = np.sqrt(np.sum(np.power(current[current.shape[0]-1,:], 2)))

        #time
        #all
        data[route-1,64] = size / 60
        #desceleration time and acceleration time
        data[route-1,65] = np.sum(acceleration > 0) / data[route-1,64]
        data[route-1,66] = np.sum(acceleration < 0) / data[route-1,64]
        #time of big normalacceleration
        data[route-1,67] = np.sum(normalacceleration > 3) / data[route-1,64]
        
        #number of turn
        data[route-1,68] = 0
        for iter in range(2,size-4):
            if normalacceleration[iter] > 3 and normalacceleration[iter] == normalacceleration[iter-2:iter+3].max():
                data[route-1,68] += 1
                
        #number of stops
        data[route-1,69] = 0
        #number of long stop > 10 seconds
        data[route-1,70] = 0
        #time of stops
        data[route-1,71] = 0
        
        tmp = 0
        for iter in range(2,size-1):
            if speed[iter] < 0.1:
                tmp += 1
                if tmp == 10:
                    data[route-1,70] += 1
                data[route-1,71] += 1                
                if speed[iter-1] != 0:
                    data[route-1,69] += 1
            else:
                tmp = 0
        
        #mean time of stop
        if data[route-1,69] != 0:
            data[route-1,72] = data[route-1,71] / data[route-1,69]
        else:
            data[route-1,72] = 0
        
        #stop per km
        if data[route-1,63] != 0:        
            data[route-1,73] = data[route-1,69] / data[route-1,63]
        else:
            data[route-1,73] = 0
            
        #long stop per km
        if data[route-1,63] != 0:
            data[route-1,74] = data[route-1,70] / data[route-1,63]
            
        #stop per minute
        if data[route-1,64] != 0:
            data[route-1,75] = data[route-1,69] / (60 * data[route-1,64])    
        else:
            data[route-1,75] = 0
            
        #proport to energy
        tmp = np.power(speed[1:], 2)-np.power(speed[:speed.size-1],2)
        data[route-1,76] = np.sum(tmp[tmp > 0])
        
    result = np.vstack((result, data))
    countdone += 1
    
#saving
joblib.dump(result, 'features77.pkl')
joblib.dump(drivernames, 'drivernames.pkl')
