from matplotlib import pyplot as plt
import pickle

def set_font(fontsize=18, font="Times New Roman"):
    '''Sets font and fontsize for all matplotlib plots.'''
    rc = {"font.size" : fontsize,
    "font.family" : font,
    "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)

def pickle_load(filepath):
    '''Loads pickled data.'''
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def pickle_save(filepath, data, protocol=2):
    '''Saves data as pickle.'''
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)

def check_array_memory(x):
    '''Prints array memory size, automatically rounds to appropriate memory unit.'''
    array_size = x.nbytes
    if array_size*1E-6 < 1:
        print('Array Size: ' + str(round(array_size*1E-3, 3)) + ' KB')
    elif array_size*1E-9 < 1:
        print('Array Size: ' + str(round(array_size*1E-6, 3)) + ' MB')
    else:
        print('Array Size: ' + str(round(array_size*1E-9, 3)) + ' GB')