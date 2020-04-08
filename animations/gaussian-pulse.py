import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
#fig.set_dpi(100)
ax1 = fig.add_subplot(1,1,1)


def gaussian(t,x):
    c=-3 #propagation speed
    mu=0 #mean
    sigma=1 #standard deviation
    Norm=np.sqrt(2.0*np.pi*sigma*sigma)
    return (1.0/Norm)*np.exp( -(x-mu+c*t)*(x-mu+c*t)/ (2.0*sigma*sigma) )



#a video is a succesion of images
N=1000 #number of points per image
x_max=+10.0 #maximum point of the space
x_min=-10.0 #minimum point of the space
hx=(x_max-x_min)/N #space-discretization step
n_images=100 #number de imagenes

c=1 #speed

Nt=n_images #number of times, 
#equivalently total number of images to be used in the video

t_min=0 #initial time
t_max=5 #final time
ht=(t_max-t_min)/(1.0*Nt) #time-discretization step

#define two empty vectors
x=np.empty([N], dtype=float)
y=np.empty([N], dtype=float)

time=t_min 

#x_min is always the starter point to swep the space
x[ 0 ]=x_min
for j in range(1,N): 
    x[j]=x[j-1]+hx 
    


def animate(i):

    global N,time

    #swep the whole space
    for j in range(0,N): 
        y [ j ]=gaussian(time,x[ j ] )


    time=time+ht #increase time

    ax1.clear() #delete previous image before plot next
    plt.plot(x,y,color="blue")
    plt.title("Gaussian Pulse")
    plt.ylim([0.0,0.6]) #set y range
    plt.xlim([-10.0,10.0]) #set xrange
	


#frames: total number of images to be used in the video
anim = animation.FuncAnimation(fig, animate, frames=n_images, interval=20)

#save in an external mp4 video file
#fps: frames per second
anim.save('pulse.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

