import numpy as np
import timeit
from tfrbm.bbrbm import BBRBM
# import matplotlib.pyplot as plt                                                                                                                    
bm = BBRBM(n_visible=36,n_hidden= 56)
bm.image_height = 6
f = open("6x6.txt")
dataset = np.array([[int(data) for data in f.read() if data in "01"]])
err = bm.fit(dataset, n_epoches=50, learning_rate = 1000, decay = 0.01, epochs_to_test = 1)

import time

start_time = time.time()
bm.sess.run(bm.hidden_p, feed_dict={bm.x: dataset})
hidden_elapsed_time = time.time() - start_time
print ("Time to calculate hidden nodes:",hidden_elapsed_time)

start_time = time.time()
bm.sess.run(bm.visible_recon_p, feed_dict={bm.x: dataset})
visible_elapsed_time = time.time() - start_time - hidden_elapsed_time
print ("Time to reconstruct visible nodes:",visible_elapsed_time)