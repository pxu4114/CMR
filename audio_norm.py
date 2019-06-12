import numpy as np
import pdb

file = np.load('new_data/f8k_precomp/train_aud_unnormed.npy')
min = np.min(file,axis = 1)
max = np.max(file,axis = 1)
min = min.reshape((30000,1))
max = max.reshape((30000,1))
numerator = np.subtract(file, min)
denominator = np.subtract(max,min)

final = np.divide(numerator,denominator)
np.save('new_data/f8k_precomp/train_aud.npy',final)


file = np.load('new_data/f8k_precomp/test_aud_unnormed.npy')
min = np.min(file,axis = 1)
max = np.max(file,axis = 1)
min = min.reshape((5000,1))
max = max.reshape((5000,1))
numerator = np.subtract(file, min)
denominator = np.subtract(max,min)

final = np.divide(numerator,denominator)
np.save('new_data/f8k_precomp/test_aud.npy',final)
pdb.set_trace()