import numpy as np
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
import csv

class Julia:
    def __init__(self, M, maxItr, c_r, c_i):
        self.M = M
        self.maxItr = maxItr
        self.c_r = c_r
        self.c_i = c_i
        self.c = c_r + 1j*c_i

    def Calc(self, re, im): # re, im: Real number 
        z_r = re
        z_i = im
        for n in range(self.maxItr):
            zr2 = z_r**2
            zi2 = z_i**2
            if zr2 + zi2 > self.M:
                return n
            z_r_ = zr2 - zi2 + self.c_r
            z_i_ = 2*z_r*z_i + self.c_i
            z_r, z_i = z_r_, z_i_
        return self.maxItr

    def Calculate(self, z0): # z0: Complex number
        z = z0
        for n in range(self.maxItr):
            az = abs(z)
            if az > self.M:
                return n
            z = z*z + self.c
        return self.maxItr

@jit
def julia_set(c_r,c_i,xmin,xmax,ymin,ymax,xpix,ypix,maxItr):
    M = 2 ** 40
    J = Julia(M, maxItr, c_r, c_i)
    re = np.linspace(xmin, xmax, xpix)
    im = np.linspace(ymin, ymax, ypix)
    z  = np.empty((xpix,ypix))
    for i in tqdm(range(xpix)):
        for j in range(ypix):
            z[j][i] = J.Calc(re[i], im[j])
            #z[j][i] = J.Calculate(re[i] + 1j*im[j])
    return re, im, z

def julia_image(c_r,c_i,x,y,radius,width=10,height=10,dpi=72,maxItr=80):
    xpix, ypix = dpi * width, dpi * height
    xmin, xmax = x - radius, x + radius
    ymin, ymax = y - radius, y + radius
    re,im,z = julia_set(c_r,c_i,xmin,xmax,ymin,ymax,xpix,ypix,maxItr)
    return re, im, np.array(z)

def julia_plot(data,geom,gamma,cmap,normz,title,figfile):
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
    project = 'julia00A'
    csvfile, figfile,txtfile = project+'.csv', project+'.png', project+'.txt'
    u = input('(1)Calculate,(2)Display : ')
    if u=='1':
        radius = 1.4
        x_o = 0
        y_o = 0
        c_r =  0.288
        c_i =  0.4836
        maxItr = 1024
        width = 10
        height = 10
        dpi = 220
        gamma = 0.6
        x,y,Z = julia_image(c_r,c_i,x_o,y_o,radius,width,height,dpi,maxItr)
        np.savetxt(csvfile, Z)
        with open( txtfile, 'w', encoding='utf-8') as f:
            f.write('c_real,c_imag,radius,o_x,o_y,maxItr,width,height,dpi,gamma,normalize\n')
            f.write('{:f},{:f},{:f},{:f},{:f},{:d},{:d},{:d},{:d},{:f},{:s}'.\
                    format(c_r,c_i,radius,x_o,y_o,maxItr,width,height,dpi,gamma,'Power'))
    else:
        Z = np.loadtxt(csvfile)
        with open( txtfile, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
            c_r, c_i = float(row[0]), float(row[1])
            radius = float(row[2])
            x_o, y_o = float(row[3]), float(row[4])
            maxItr = int(row[5])
            width, height = int(row[6]), int(row[7])
            dpi = int(row[8])
            gamma = float(row[9])
            normz = row[10]
        xmin, xmax = x_o - radius, x_o + radius
        ymin, ymax = y_o - radius, y_o + radius
        x = np.linspace(xmin, xmax, dpi*width)
        y = np.linspace(ymin, ymax, dpi*height)
        cmap = 'hot'
        title = 'c=({},{}),o=({},{}),r={},maxItr={}'.format(c_r,c_i,x_o,y_o,radius,maxItr)
        normz = 'Log'
        data = (x,y,Z)
        geom = (xmin,xmax,ymin,ymax,width,height,dpi)
        julia_plot(data,geom,gamma,cmap,normz,title,figfile)
