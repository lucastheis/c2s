#%%cython
# Circumvents a bug(?) in cython:
# http://stackoverflow.com/a/13976504
# STUFF = "Hi"


import numpy as np
cimport numpy as np
cimport cython

# Do not check for index errors
@cython.boundscheck(False)
# Do not enable negativ indices
@cython.wraparound(False)
# Use native c division
@cython.cdivision(True)
def roc(np.ndarray[double, ndim=1] positives, np.ndarray[double, ndim=1] negatives):
    """calculate ROC score for given values of positive and negative
    distribution"""
    cdef np.ndarray[double, ndim=1] all_values = np.hstack([positives, negatives])
    all_values = np.sort(all_values)[::-1]
    cdef np.ndarray[double, ndim=1] sorted_positives = np.sort(positives)[::-1]
    cdef np.ndarray[double, ndim=1] sorted_negatives = np.sort(negatives)[::-1]
    cdef np.ndarray[double, ndim=1] false_positive_rates = np.zeros(len(all_values)+1)
    cdef np.ndarray[double, ndim=1] hit_rates = np.zeros(len(all_values)+1)
    cdef int true_positive_count = 0
    cdef int false_positive_count = 0
    cdef int positive_count = len(positives)
    cdef int negative_count = len(negatives)
    cdef int i
    cdef float theta
    for i in range(len(all_values)):
        theta = all_values[i]
        while true_positive_count < positive_count and sorted_positives[true_positive_count] >= theta:
            true_positive_count += 1
            if true_positive_count >= positive_count:
                break
        while false_positive_count < negative_count and sorted_negatives[false_positive_count] >= theta:
            false_positive_count += 1
            if false_positive_count < negative_count:
                break
        false_positive_rates[i+1] = float(false_positive_count) / negative_count
        hit_rates[i+1] = float(true_positive_count) / positive_count
        #hit_rates.append((positives>=theta).mean())
    #false_positive_rates.append(1.0)
    #hit_rates.append(1.0)
    auc = np.trapz(hit_rates, false_positive_rates)
    return auc, hit_rates, false_positive_rates
