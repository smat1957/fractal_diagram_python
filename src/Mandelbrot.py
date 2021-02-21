# https://nepia01.blogspot.com/2017/07/python.html
import numpy as np
from numba import jit
from tqdm import tqdm
 
@jit
def mandelbrot(z,maxiter,horizon,log_horizon):
    c = z
    for n in range(maxiter):
        az = abs(z)
        if az > horizon:
            return n - np.log(np.log(az))/np.log(2) + log_horizon
        z = z*z + c
    return 0
 
@jit
def mandelbrot_set(xmin,xmax,ymin,ymax,width,height,maxiter):
    horizon = 2 ** 40
    log_horizon = np.log(np.log(horizon))/np.log(2)
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width,height))
    for i in tqdm(range(width)):
        for j in range(height):
            n3[i,j] = mandelbrot(r1[i] + 1j*r2[j],maxiter,horizon, log_horizon)
    return (r1,r2,n3)
 
from matplotlib import pyplot as plt
from matplotlib import colors
  
def mandelbrot_image(x,y,radius,width=10,height=10,maxiter=80,cmap='jet',gamma=0.3):
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    xmin = x - radius
    xmax = x + radius
    ymin = y - radius
    ymax = y + radius
    x,y,z = mandelbrot_set(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)
     
    fig, ax = plt.subplots(figsize=(width, height),dpi=72)
    ticks = np.arange(0,img_width,3*dpi)
    x_ticks = xmin + (xmax-xmin)*ticks/img_width
    plt.xticks(ticks, x_ticks)
    y_ticks = ymin + (ymax-ymin)*ticks/img_width
    plt.yticks(ticks, y_ticks)
    ax.set_title(cmap)
     
    norm = colors.PowerNorm(gamma)
    cax = ax.imshow(z.T,cmap=cmap,origin='lower',norm=norm)
    fig.colorbar(cax, shrink=0.82)
    plt.savefig('fig000.png')
    plt.show()

mandelbrot_image(-0.75, 0, 1.5, cmap='hot', gamma=0.4)
#mandelbrot_image(0.2501,1.6e-6,1e-8,maxiter=10000)
#mandelbrot_image(0.3007100003,0.02006000303,1e-10,maxiter=2048,cmap='hot',gamma=0.4)
#mandelbrot_image(-0.74958245,0.0300396,1e-6,maxiter=30000,cmap='copper')
#mandelbrot_image(-0.74997501,-0.00999551,1e-8,maxiter=100000,cmap='flag',gamma=0.98)
