import math
import matplotlib.pyplot as plt

class koch():
    def __init__( self,p1, p2, gene=0 ):
        self.result=[]
        self.gene = gene
        self.p1 = p1
        self.p2 = p2
        self.p3 = [(2*self.p1[0] + self.p2[0]) / 3,(2*self.p1[1] + self.p2[1]) / 3]
        self.p4 = [(self.p1[0] + 2*self.p2[0]) / 3,(self.p1[1] + 2*self.p2[1]) / 3]
        xm = [0,0]
        xm[0] = (self.p2[0] + self.p1[0]) / 2
        xm[1] = (self.p2[1] + self.p1[1]) / 2
        l = math.sqrt((self.p2[0]-self.p1[0])**2+(self.p2[1]-self.p1[1])**2)/3
        h = l*math.sqrt(3)/2
        if self.p2[0]==self.p1[0]:
            if self.p1[1] < self.p2[1]:
                h=-h
            self.p5 = [xm[0]-h,xm[1]]
        else:
            beta = math.atan((self.p2[1]-self.p1[1])/(self.p2[0]-self.p1[0]))
            alpha = math.pi/2-beta
            if self.p1[0] > self.p2[0]:                
                self.p5 = [xm[0]+h*math.cos(alpha),xm[1]-h*math.sin(alpha)]
            else:
                self.p5 = [xm[0]-h*math.cos(alpha),xm[1]+h*math.sin(alpha)]
        self.generate()
        
    def generate( self ):
        if self.gene > 0:
            k1 = koch( self.p1, self.p3, self.gene-1 )
            k2 = koch( self.p3, self.p5, self.gene-1 )
            k3 = koch( self.p5, self.p4, self.gene-1 )
            k4 = koch( self.p4, self.p2, self.gene-1 )
            self.result.extend(k1.getResult())
            self.result.extend(k2.getResult())
            self.result.extend(k3.getResult())
            self.result.extend(k4.getResult())
        else:
            #self.getPrint()
            return self.result.extend( self.getCoordinates() )
    
    def getCoordinates( self ):
        return [self.p1, self.p3, self.p5, self.p4, self.p2]

    def getResult( self ):
        return self.result

    def getPrint(self):
        print( 'x:',self.p1[0],',y:',self.p1[1] )
        print( 'x:',self.p3[0],',y:',self.p3[1] )
        print( 'x:',self.p5[0],',y:',self.p5[1] )
        print( 'x:',self.p4[0],',y:',self.p4[1] )
        print( 'x:',self.p2[0],',y:',self.p2[1] )

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.35, 0.9)
    
    gene=2
    x,y=[],[]
    
    p1, p2 = [0,0], [1,0]
    k = koch(p1, p2, gene)
    k0 = k.getResult()
    for j in range( len(k0) ):
        x.append(k0[j][0])
        y.append(k0[j][1])

    plt.plot(x,y)
    fig.savefig('figkoch1.png')
    fig.show()
