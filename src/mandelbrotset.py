'''
mandelbrotset.py
Draw a Mandelbrot set
Using "Escape time algorithm" from:
http://en.wikipedia.org/wiki/Mandelbrot_set#Computer_drawings
'''
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt

# Subset of the complex plane we are considering
x0, x1 = -2.5, 1  # x0, x1 = -0.75, -0.749
y0, y1 = -1, 1    # y0, y1 = 0.040, 0.041

@jit
def mandelbrot_set(n, max_iterations):
    image = initialize_image(n, n)    
    # Generate a set of equally spaced points
    #  in the region above
    dx = (x1-x0)/(n-1)
    dy = (y1-y0)/(n-1)
    x_coords = [x0 + i*dx for i in range(n)]
    y_coords = [y0 + i*dy for i in range(n)]
    #
    for i, x in tqdm( enumerate(x_coords) ):
        for k, y in enumerate(y_coords):
            z1 = complex(0, 0)
            iteration = 0
            c = complex(x, y)
            while (abs(z1) < 2  and iteration < max_iterations):
                z1 = z1**2 + c
                iteration += 1
            image[k][i] = iteration
    return image

def initialize_image(x_p, y_p):
    image = []
    for i in range(y_p):
        x_colors = []
        for j in range(x_p):
            x_colors.append(0)
        image.append(x_colors)
    return image

def draw_mandelbrot(n, max_iterations):
    image = mandelbrot_set(n, max_iterations)
    clmap = 'hot'    # clmap = 'Greys_r' or 'PuBu_r' or 'hot' etc
    # 画素の補間（Nearest neighbor,Bilinear,Bicubic）
    plt.imshow(image, origin='lower', extent=(x0, x1, y0, y1),
               cmap=clmap, interpolation='nearest')
    plt.savefig('mandelbrotset00.png')
    plt.show()

if __name__ == '__main__':
    n = int( input('Enter the image size nxn points (100 to 1000) : ') )
    max_iter = int( input('Enter the max number of iteration (100 to 10000) : ') )
    draw_mandelbrot(n, max_iter)
    # draw_mandelbrot(n=1000, max_iteration=100)
