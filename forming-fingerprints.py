from microphone import record_audio, play_audio

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