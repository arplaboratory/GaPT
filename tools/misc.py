import math
import numpy as np
import copy
from tqdm import tqdm
import time


##############################
#       TIME                 #
##############################

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        elapsed_time = time.time() - self.tstart
        print('Elapsed: %s' % elapsed_time)


##############################
#   SORTING                  #
##############################

def distance_3d(v1: np.ndarray, v2: np.ndarray):
    dist = np.matmul((v1 - v2).T, (v1 - v2))
    return dist


def sort_3d(inputList):
    start_idx = 0
    inputArray = copy.deepcopy(inputList)
    output_array = []
    output_array.append(0)

    # SORT BASED ON LAST POINT
    for i in tqdm(range(1, inputArray.shape[0]), desc="Sorting dataset", leave=False):
        start_idx = output_array[i - 1]
        start_value = inputArray[start_idx]
        min_cost = 999999999.9
        min_idx = -1
        for j in range(inputArray.shape[0]):
            if start_idx != j and not (j in output_array):
                curr_dist = distance_3d(start_value, inputArray[j])
                if min_cost > curr_dist:
                    min_idx = j
                    min_cost = curr_dist
        output_array.append(min_idx)

    return output_array


def desorting_indices_calc(sorting_indexes):
    sorting_indexes = np.asarray(sorting_indexes)
    # Get the rollback indexes
    rollback_indexes = np.empty_like(sorting_indexes)
    rollback_indexes[sorting_indexes] = np.arange(sorting_indexes.size)
    return rollback_indexes


def predict_get_two_index(datasets, start, end):
    start_idx = -1
    min_start_cost = 999999999.9
    end_idx = -1
    min_end_cost = 999999999.9
    for i in range(datasets.shape[0]):
        curr_dist = distance_3d(datasets[i], start)
        if min_start_cost > curr_dist:
            start_idx = i
            min_start_cost = curr_dist
        curr_dist = distance_3d(datasets[i], end)
        if min_end_cost > curr_dist:
            end_idx = i
            min_end_cost = curr_dist
    if end_idx < start_idx:
        return end_idx, start_idx
    return start_idx, end_idx


##############################
#   SEARCH                  #
##############################

def siso_binary_search(array, val):
    low = 0
    high = len(array) - 1
    mid = 0
    # edge case: value of smaller than min or larger than max
    if array[low] >= val:
        return 0
    if array[high] <= val:
        return high
    while low <= high:
        mid = math.floor((low+high)/2)
        # value is in interval from previous to current element
        if array[mid - 1] <= val <= array[mid]:
            if abs(val - array[mid - 1]) < abs(val - array[mid]):
                return mid - 1
            else:
                return mid
        elif array[mid] < val:
            low = mid + 1
        else:
            high = mid - 1
    return -1
