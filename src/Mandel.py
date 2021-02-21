import numpy as np
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import csv

class Mandelbrot:
    def __init__(self, M, maxItr):
        self.M = M
        self.maxItr = maxItr
        
    def Calc(self, re, im): # re, im: 2d Array 
        x = 0.0
        y = 0.0
        for n in range(self.maxItr):
            zx2 = x**2
            zy2 = y**2
            if zx2 + zy2 > self.M:
                return n
            x_ = zx2 - zy2 + re
            y_ = 2*x*y + im
            x, y = x_, y_
        return self.maxItr
    
    def Calculate(self, c): # c: complex
        # z0 = 0.0 + 1j*0.0
        z = c # z = z0**2 + c
        for n in range(self.maxItr):
            az = abs(z)
            if az > self.M:
                return n
            z = z*z + c
        return self.maxItr

@jit
def mandel_set(xmin,xmax,ymin,ymax,xpix,ypix,maxItr):
    M = 2 ** 40
    Mandel = Mandelbrot(M, maxItr)
    re = np.linspace(xmin, xmax, xpix)
    im = np.linspace(ymin, ymax, ypix)
    z  = np.empty((xpix, ypix))
    for i in tqdm(range(xpix)):
        for j in range(ypix):
            z[j][i] = Mandel.Calc(re[i], im[j])
            #z[j][i] = Mandel.Calculate(re[i] + 1j*im[j])
    return re, im, z

def mandel_image(x,y,radius,width=10,height=10,dpi=72,maxItr=80):
    xpix, ypix = dpi * width, dpi * height
    xmin, xmax = x - radius, x + radius
    ymin, ymax = y - radius, y + radius
    re, im, z = mandel_set(xmin,xmax,ymin,ymax,xpix,ypix,maxItr)
    return re, im, np.array(z)

def mandel_plot(data,geom,gamma,cmap,normz,title,figfile):
    x,y,Z = data[0], data[1], data[2]
    xmin,xmax,ymin,ymax,width,height,dpi = geom[0],geom[1],geom[2],geom[3],geom[4],geom[5],geom[6]
    X, Y = np.meshgrid(x, y)
    plt.rcParams['figure.figsize'] = width,height
    plt.rcParams["figure.dpi"] = dpi
    plt.gca().set_aspect('equal', adjustable='box')
    if normz=='Power':
        norm = colors.PowerNorm(gamma)
    elif normz=='Log':
        norm = colors.LogNorm(vmin=Z.min(), vmax=Z.max())
    pcm = plt.pcolor(X, Y, Z,norm=norm,cmap=cmap)
    plt.colorbar(pcm,shrink=0.82)
    plt.title(title)
    plt.xlabel('Re', fontsize=22)
    plt.ylabel('Im', fontsize=22)
    plt.xticks(np.linspace(xmin,xmax,3), color="black")
    plt.yticks(np.linspace(ymin,ymax,3), color="black")
    plt.tick_params(length=0)
    plt.savefig(figfile)
    plt.show()

if __name__ == '__main__':
    project = 'mandel010'
    csvfile,figfile,txtfile = project+'.csv', project+'.png', project+'.txt'
    u = input('(1)Calculate,(2)Display : ')
    if u=='1':
        radius = 0.0005
        x_o = -0.75 + radius
        y_o =  0.04 + radius
        maxItr = 1024
        width,height,dpi = 10,10,72
        gamma = 0.8
        x, y, Z = mandel_image(x_o,y_o,radius,width,height,dpi,maxItr)
        np.savetxt(csvfile, Z)
        with open( txtfile, 'w', encoding='utf-8') as f:
            f.write('radius,o_x,o_y,maxItr,width,height,dpi,gamma,normalize\n')
            f.write('{:f},{:f},{:f},{:d},{:d},{:d},{:d},{:f},{:s}'.\
                    format(radius,x_o,y_o,maxItr,width,height,dpi,gamma,'Power'))
    else:
        Z = np.loadtxt(csvfile)
        with open( txtfile, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
            radius = float(row[0])
            x_o, y_o = float(row[1]), float(row[2])
            maxItr = int(row[3])
            width, height = int(row[4]), int(row[5])
            dpi = int(row[6])
            gamma = float(row[7])
            normz = row[8]
        xmin, xmax = x_o - radius, x_o + radius
        ymin, ymax = y_o - radius, y_o + radius
        x = np.linspace(xmin, xmax, dpi*width)
        y = np.linspace(ymin, ymax, dpi*height)
        cmap = 'PuBu_r'
        title = 'o=({0},{1}),r={2},maxItr={3}'.format(x_o,y_o,radius,maxItr)
        normz = 'Log'
        data = (x,y,Z)
        geom = (xmin,xmax,ymin,ymax,width,height,dpi)
        mandel_plot(data,geom,gamma,cmap,normz,title,figfile)
