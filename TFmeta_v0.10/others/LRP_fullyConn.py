import numpy
import copy
import modules

lowest = -1.0
highest = 1.0

X  = numpy.fromfile(open('testInput.txt'),dtype='float',count=4*569,sep="\t").reshape([4,569])
T  = numpy.fromfile(open('testOutput.txt'),dtype='float',count=4*42,sep="\t").reshape([4,42])

nn = modules.Network([
    modules.Linear('l1'),modules.tanh(),
    modules.Linear('l2'),modules.tanh(),
    modules.Linear('lout'),
])

Y = nn.forward(X)
S = nn.gradprop(T)**2
Y.tofile("sensitivity_pred.txt", sep='\t')
S.tofile("sensitivity.txt", sep='\t')

class Network(modules.Network):
    def relprop(self,R):
        for l in self.layers[::-1]: R = l.relprop(R)
        return R

class ReLU(modules.ReLU):
    def relprop(self,R): return R

###### Self-build: tanh ######
class tanh(modules.tanh):
    def relprop(self,R): return R

class NextLinear(modules.Linear):
    def relprop(self,R):
        V = numpy.maximum(0,self.W)
        Z = numpy.dot(self.X,V)+1e-9; S = R/Z
        C = numpy.dot(S,V.T);         R = self.X*C
        return R

class FirstLinear(modules.Linear):
    def relprop(self,R):
        W,V,U = self.W,numpy.maximum(0,self.W),numpy.minimum(0,self.W)
        X,L,H = self.X,self.X*0+lowest,self.X*0+highest

        Z = numpy.dot(X,W)-numpy.dot(L,V)-numpy.dot(H,U)+1e-9; S = R/Z
        R = X*numpy.dot(S,W.T)-L*numpy.dot(S,V.T)-H*numpy.dot(S,U.T)
        return R

nn = Network([
    FirstLinear('l1'),tanh(),
    NextLinear('l2'),tanh(),
    NextLinear('lout'),tanh(),
])

Y = nn.forward(X)
D = nn.relprop(Y*T)
Y.tofile("taylor_pred.txt", sep='\t')
D.tofile("taylor.txt", sep='\t')
