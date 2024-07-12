import numpy as np

# Step 8: Generating random clips
def generate_random_clips(samples, clip_length, num_clips):
    total_samples = len(samples)
    clip_samples = clip_length * 44100
    clips = []
    for _ in range(num_clips):
        start = np.random.randint(0, total_samples - clip_samples)
        clip = samples[start:start + clip_samples]
        clips.append(clip)
    return clips
