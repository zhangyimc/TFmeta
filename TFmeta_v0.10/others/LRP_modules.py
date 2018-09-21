import numpy

# -------------------------
# Feed-forward network
# -------------------------
class Network:

	def __init__(self,layers):
		self.layers = layers

	def forward(self,Z):
		for l in self.layers: Z = l.forward(Z)
		return Z

	def gradprop(self,DZ):
		for l in self.layers[::-1]: DZ = l.gradprop(DZ)
		return DZ

# -------------------------
# ReLU activation layer
# -------------------------
class ReLU:

	def forward(self,X):
		self.Z = X>0
		return X*self.Z

	def gradprop(self,DY):
		return DY*self.Z

# -------------------------
# Self-build: tanh activation layer
# -------------------------
class tanh:

	def forward(self,X):
		self.Z = numpy.tanh(X)
		return self.Z

	def gradprop(self,DY):
		return DY*(1.0-self.Z**2)

# -------------------------
# Fully-connected layer
# -------------------------
class Linear:

	def __init__(self,name):
		self.W = numpy.loadtxt(name+'-W.txt')
		self.B = numpy.loadtxt(name+'-B.txt')

	def forward(self,X):
		self.X = X
		return numpy.dot(self.X,self.W)+self.B

	def gradprop(self,DY):
		self.DY = DY
		return numpy.dot(self.DY,self.W.T)
