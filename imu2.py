#imu homework
#Teddy Robbins 2021

import numpy as np
import scipy.integrate as it
import matplotlib.pyplot as plt 
import pandas as pd

#Read in data to pandas df
df1 = pd.read_csv('Accelerometer.csv')
df2 = pd.read_csv('Gyroscope.csv')
timeCol = df1.columns[0]
axCol = df1.columns[1]
ayCol = df1.columns[2]
wzCol = df2.columns[3]

#Get the data we really want into a np array
time = df1[timeCol].to_numpy()[:375]
ax = df1[axCol].to_numpy()[:375]
ay = df1[ayCol].to_numpy()[:375]
wz = df2[wzCol].to_numpy()[:375]

#Resize arrays to match lengths
diff = time.shape[0]-wz.shape[0]

if diff>0:
	time = time[diff:]
	ax = ax[diff:]
	ay = ay[diff:]
elif diff<0:
	wz=wz[-diff:]

#Integrate omega to get theta
thetaz = it.cumtrapz(wz, x=time)
thetaz = np.insert(thetaz,0,0)

#Integrate accelerations twice to get rb
rbx = it.cumtrapz(it.cumtrapz(ax,x=time),x=time[1:])
rby = it.cumtrapz(it.cumtrapz(ay,x=time),x=time[1:])
rb = np.array([rbx,rby]).T

#Convert to local-frame
rl = np.zeros(rb.shape)

for i in range(2,thetaz.shape[0]):
	#Set up DCM
	DCM = np.array([[np.cos(thetaz[i]),np.sin(thetaz[i])],[-np.sin(thetaz[i]),np.cos(thetaz[i])]])
	#Multiply body frame by DCM to get local frame
	rl[i-2]=np.matmul(rb[i-2],DCM)

print('Final local Position: %s'%(str(rl[-1])))
print('Error: %s'%(str(rl[-1]-np.array([3.05,3.05]))))

#Show data
fig1 = plt.figure()
plt.subplot(3,1,1)
plt.title('x Position (in body frame) vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('x Position (m)')
plt.plot(time[2:], rbx.T)
plt.subplot(3,1,2)
plt.title('y Position (in body frame) vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('y Position (m)')
plt.plot(time[2:], rby.T)
plt.subplot(3,1,3)
plt.title('Heading vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Heading (rad)')
plt.plot(time, thetaz)
fig1.tight_layout(pad=0.5)

fig2 = plt.figure()
plt.subplot(2,1,1)
plt.title('x Position (in local frame) vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('x Position (m)')
plt.plot(time[2:], rl.T[0])
plt.subplot(2,1,2)
plt.title('y Position (in local frame) vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('y Position (m)')
plt.plot(time[2:], rl.T[1])
fig2.tight_layout(pad=0.5)

fig3 = plt.figure()
plt.title('Object Path in Local Frame')
plt.xlabel('x Position (m)')
plt.ylabel('y Position (m)')
plt.plot(rl.T[0],rl.T[1])
plt.plot(3.05,3.05,'ro')
plt.show()
