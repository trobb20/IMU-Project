#imu homework
#Teddy Robbins 2021

import numpy as np
import scipy.integrate as it
import matplotlib.pyplot as plt 
import pandas as pd

#Read in data to pandas df
df = pd.read_csv('Raw.csv')
timeCol = df.columns[0]
absAccelCol = df.columns[2]

#Get the data we really want into a np array
time = df[timeCol].to_numpy()
accel = df[absAccelCol].to_numpy()

#Integrate twice to get position
velocity = it.cumtrapz(accel,x=time)
velocity = np.insert(velocity,0,0)
position = it.cumtrapz(velocity,x=time)
position = np.insert(position,0,0)

#Plot
figure = plt.figure()
plt.subplot(3,1,1)
plt.title('Acceleration vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s2)')
plt.plot(time,accel)
plt.subplot(3,1,2)
plt.title('Velocity vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.plot(time,velocity)
plt.subplot(3,1,3)
plt.title('Position vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.plot(time,position)
figure.tight_layout(pad=0.5)
plt.show()

#Calculate final position
final_m = position[-1]
final_ft = 39.37/12 * final_m
print('Final position was: %f meters or %f feet.'%(final_m,final_ft))

'''Final position was: 7.252458 meters or 23.794107 feet.'''