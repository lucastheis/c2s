#%%cython
# Circumvents a bug(?) in cython:
# http://stackoverflow.com/a/13976504
# STUFF = "Hi"


import numpy as np
cimport numpy as np
cimport cython


#Do not check for index errors
@cython.boundscheck(False)
#Do not enable negativ indices
@cython.wraparound(False)
#Use native c division
@cython.cdivision(True)
def real_ROC(image, fixation_data, int judd=0):
    fixations_orig = np.zeros_like(image)
    fixations_orig[fixation_data] = 1.0
    image_1d = image.flatten()
    cdef np.ndarray[double, ndim=1] fixations = fixations_orig.flatten()
    inds = image_1d.argsort()
    #image_1d = image_1d[inds]
    fixations = fixations[inds]
    cdef int i
    cdef int N = image_1d.shape[0]
    cdef int fix_count = fixations.sum()
    cdef int false_count = N-fix_count
    cdef int correct_count = 0
    cdef int false_positive_count = 0
    cdef int length
    if judd:
        length = fix_count+2
    else:
        length = N+1
    cdef np.ndarray[double, ndim=1] precs = np.zeros(length)
    cdef np.ndarray[double, ndim=1] false_positives = np.zeros(length)
    for i in range(N):
        #print fixations[N-i-1],
        #print image_1d[N-i-1]
        if fixations[N-i-1]:
            correct_count += 1
            if judd:
                precs[correct_count] = float(correct_count)/fix_count
                false_positives[correct_count] = float(false_positive_count)/false_count
        else:
            false_positive_count += 1
        if not judd:
            precs[i+1] = float(correct_count)/fix_count
            false_positives[i+1] = float(false_positive_count)/false_count
        #print false_positives[i+1]
    precs[length-1] = 1.0
    false_positives[length-1] = 1.0
    aoc = np.trapz(precs, false_positives)
    return aoc, precs, false_positives
