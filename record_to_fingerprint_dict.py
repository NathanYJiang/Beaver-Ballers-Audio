#to stop deprecation errors (doesn't work)
import warnings
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio
from typing import Union, Callable, Tuple, List
from pathlib import Path
#from scipy.ndimage.filters import maximum_filter
#from scipy.ndimage.morphology import generate_binary_structure
#from scipy.ndimage.morphology import iterate_structure
from microphone import record_audio

#to display plots
import matplotlib
matplotlib.use('TkAgg')

'''
functions are defined and called at the end
'''

def record_and_save():
    listen_time = 10 # seconds
    frames, sample_rate = record_audio(listen_time)

    return (frames, sample_rate)

##frames, sample_rate = record_and_save()

#samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])
#array_to_save = np.hstack((sample_rate, samples))



#needs input regarding
#recorded_audio -> intial audio sample np.array
#sampling_rate -> sampling rate used for converting analog signal to np.array of samples
def take_input():
    recorded_audio_str = input("Enter np.array of audio samples: ")
    sampling_rate_str = input("Enter sampling rate: ")
    
    recorded_audio = np.fromstring(recorded_audio_str.strip("[]"), sep=',')
    sampling_rate = int(sampling_rate_str)
    return (recorded_audio, sampling_rate)

#calculates and returns values for spectrogram
def calc_spectrogram(recorded_audio, sampling_rate):
    import matplotlib.mlab as mlab
    fig, ax = plt.subplots()

    #calculatates spectogram
    S, freqs, times, im = ax.specgram(
        recorded_audio,
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap= 4096 // 2,
        mode='magnitude',
        scale="dB"
    )
    #fig.colorbar(im)
    #ax.set_xlabel("Time [seconds]")
    #ax.set_ylabel("Frequency (Hz)")
    #ax.set_title("Spectrogram of Recording")
    return (S, freqs, times, im)


from numba import njit

# `@njit` "decorates" the `_peaks` function. This tells Numba to
# compile this function using the "low level virtual machine" (LLVM)
# compiler. The resulting object is a Python function that, when called,
# executes optimized machine code instead of the Python code
# 
# The code used in _peaks adheres strictly to the subset of Python and
# NumPy that is supported by Numba's jit. This is a requirement in order
# for Numba to know how to compile this function to more efficient
# instructions for the machine to execute
@njit
def _peaks(
    data_2d: np.ndarray, nbrhd_row_offsets: np.ndarray, nbrhd_col_offsets: np.ndarray, amp_min: float
) -> List[Tuple[int, int]]:
    """
    A Numba-optimized 2-D peak-finding algorithm.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    nbrhd_row_offsets : numpy.ndarray, shape-(N,)
        The row-index offsets used to traverse the local neighborhood.
        
        E.g., given the row/col-offsets (dr, dc), the element at 
        index (r+dr, c+dc) will reside in the neighborhood centered at (r, c).
    
    nbrhd_col_offsets : numpy.ndarray, shape-(N,)
        The col-index offsets used to traverse the local neighborhood. See
        `nbrhd_row_offsets` for more details.
        
    amp_min : float
        All amplitudes equal to or below this value are excluded from being
        local peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location, returned in 
        column-major order
    """
    peaks = []  # stores the (row, col) locations of all the local peaks

    # Iterating over each element in the the 2-D data 
    # in column-major ordering
    #
    # We want to see if there is a local peak located at
    # row=r, col=c
    for c, r in np.ndindex(*data_2d.shape[::-1]):
        #print(data_2d[r,c])

        if data_2d[r, c] <= amp_min:
            # The amplitude falls beneath the minimum threshold
            # thus this can't be a peak.
            continue
        
        # Iterating over the neighborhood centered on (r, c) to see
        # if (r, c) is associated with the largest value in that
        # neighborhood.
        #
        # dr: offset from r to visit neighbor
        # dc: offset from c to visit neighbor
        for dr, dc in zip(nbrhd_row_offsets, nbrhd_col_offsets):
            if dr == 0 and dc == 0:
                # This would compare (r, c) with itself.. skip!
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                # neighbor falls outside of boundary.. skip!
                continue

            if not (0 <= c + dc < data_2d.shape[1]):
                # neighbor falls outside of boundary.. skip!
                continue

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                # One of the amplitudes within the neighborhood
                # is larger, thus data_2d[r, c] cannot be a peak
                break
        else:
            # if we did not break from the for-loop then (r, c) is a local peak
            peaks.append((r, c))
    return peaks




# `local_peak_locations` is responsible for taking in the boolean mask `neighborhood`
# and converting it to a form that can be used by `_peaks`. This "outer" code is 
# not compatible with Numba which is why we end up using two functions:
# `local_peak_locations` does some initial pre-processing that is not compatible with
# Numba, and then it calls `_peaks` which contains all of the jit-compatible code
def local_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray, amp_min: float):
    """
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min`.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected
    
    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location, returned
        in column-major ordering.
    
    Notes
    -----
    The local peaks are returned in column-major order, meaning that we 
    iterate over all nbrhd_row_offsets in a given column of `data_2d` in search for
    local peaks, and then move to the next column.
    """

    # We always want our neighborhood to have an odd number
    # of nbrhd_row_offsets and nbrhd_col_offsets so that it has a distinct center element
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1
    
    # Find the indices of the 2D neighborhood where the 
    # values were `True`
    #
    # E.g. (row[i], col[i]) stores the row-col index for
    # the ith True value in the neighborhood (going in row-major order)
    nbrhd_row_indices, nbrhd_col_indices = np.where(neighborhood)
    

    # Shift the neighbor indices so that the center element resides 
    # at coordinate (0, 0) and that the center's neighbors are represented
    # by "offsets" from this center element.
    #
    # E.g., the neighbor above the center will has the offset (-1, 0), and 
    # the neighbor to the right of the center will have the offset (0, 1).
    nbrhd_row_offsets = nbrhd_row_indices - neighborhood.shape[0] // 2
    nbrhd_col_offsets = nbrhd_col_indices - neighborhood.shape[1] // 2

    return _peaks(data_2d, nbrhd_row_offsets, nbrhd_col_offsets, amp_min=amp_min)


def local_peaks_mask(data: np.ndarray, cutoff: float) -> np.ndarray:
    """Find local peaks in a 2D array of data and return a 2D array
    that is 1 wherever there is a peak and 0 where there is not.

    Parameters
    ----------
    data : numpy.ndarray, shape-(H, W)

    cutoff : float
         A threshold value that distinguishes background from foreground

    Returns
    -------
    Binary indicator, of the same shape as `data`. The value of
    1 indicates a local peak."""
    # Generate a rank-2, connectivity-2 neighborhood array
    # We will not use `iterate_structure` in this example
    neighborhood_array = generate_binary_structure(rank=2, connectivity=2)

    # Use that neighborhood to find the local peaks in `data`.
    # Pass `cutoff` as `amp_min` to `local_peak_locations`.
    peak_locations = local_peak_locations(data_2d = data, neighborhood=neighborhood_array,
                                          amp_min=cutoff)
    #print(peak_locations)

    # Turns the list of (row, col) peak locations into a shape-(N_peak, 2) array
    # Save the result to the variable `peak_locations`
    peak_locations = np.array(peak_locations)

    # create a boolean mask of zeros with the same shape as `data`
    mask = np.zeros(data.shape, dtype=bool)

    # populate the local peaks with `1`
    mask[peak_locations[:, 0], peak_locations[:, 1]] = 1
    return mask


def form_fingerprints(peaks: np.ndarray, fanout=5):
    fingerprints = {}
    rows = len(peaks)
    cols = len(peaks[0])

    for row in range(rows):
        for col in range(cols):
            if peaks[row][col] == False:
                continue
            else:
                count = 0
                
                for i in range(cols-col-1): #cycles through from next col to end col major wise
                    #check if [row][i] is a match
                    i += col+1
                    if peaks[row][i] == True:
                        count += 1
                        fingerprints[row, row, i] = "songname"
                    for a in range(rows-1):
                        a += 1
                        if row + a < rows:
                            if peaks[row+a][i] == True:
                                fingerprints[row, row+a, i] = "songname"
                                count +=1
                        if row - a >= 0:
                            if peaks[row-a][i] == True:
                                fingerprints[row, row-a, i] = "songname"
                                count +=1
                        if count == fanout:
                            break
                        #check if row+a is within bound
                        #check [row+a][i]
                        #check if row-a is within bound
                        #check [row-a][i]
                        #add to counter if its a match
                        #if count == 5, break
                    if count == fanout:
                        break
    return fingerprints





'''
start of calling functions
'''
recorded_audio, sampling_rate = record_and_save()

samples = np.hstack([np.frombuffer(i, np.int16) for i in recorded_audio])
array_to_save = np.hstack((sampling_rate, samples))
np.save('/Users/manitmehta/Desktop/Cogworks', array_to_save)

#recorded_audio, sampling_rate = take_input()
S, freqs, times, im = calc_spectrogram(samples, sampling_rate)

#plots spectrogram. comment out if don't wanna plot.
fig, ax = plt.subplots()
fig.colorbar(im)
ax.set_xlabel("Time [seconds]")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Spectrogram of Recording")


peaks = local_peaks_mask(data = S, cutoff = np.percentile(S, 99.9)) #use ecdf for cutoff !

'''
print(np.max(S))
print(np.mean(S))
print(np.percentile(S, 95))
print(peaks.shape)
'''

true_indices = np.argwhere(peaks)
fingers = form_fingerprints(peaks)
print(fingers)

# Scatter plot of True values
#plt.figure(figsize=(8, 6))
plt.scatter(true_indices[:, 1], true_indices[:, 0], marker='o', color='blue', alpha=0.5)
plt.title('Scatter Plot of True Values')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.grid(True)
plt.show()

#print(mask[0:100])
#print(type(peaks))




'''

def form_fingerprints(peaks, fanout=5):
    """
    Takes peaks List[Tuple(int, int)] and fanout (number of nearest neighboring peaks)
    returns dictionary mapping
    """
    fingerprint = {}
    for m, (f_m, t_m) in enumerate(peaks):
        for i in range(1, fanout + 1):
            fingerprint[(f_m, peaks[m + i][0], peaks[m + i][1] - t_m)] = t_m
    
    return fingerprint

    
    go through each peak. cycle forward in x direction and take the 
    closest values that pop up 
    put it in dictionary

'''

 