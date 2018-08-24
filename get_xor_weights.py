import numpy as np
import timeit
from tfrbm.bbrbm import BBRBM

bm = BBRBM(n_visible=76,n_hidden=76)

#dataset = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
dataset = int(np.random.rand(76,) > 0.5).reshape(1,-1)
#import sys
#i = int(sys.argv[1])

#x = dataset[i:i+1].copy()
x = np.array(dataset[0][:76]).reshape(1,-1)
while not np.all(np.logical_and(bm.reconstruct(x), x)):

    bm.fit(dataset,n_epoches=1000)
    print np.sigmoid(bm.reconstruct(x))
    print x
weights = bm.get_weights()
w = weights[0].tolist()
b = np.concatenate([weights[1],weights[2]]).tolist()
print w
print
print b
print 
print v

