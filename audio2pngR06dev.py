import os
import sys
print(sys.executable)
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import csv
from matplotlib.colors import hsv_to_rgb
from scipy import signal, fftpack
from PIL import Image
import colorsys
from pathlib import Path
#from memory_profiler import profile
import matplotlib; matplotlib.use('Agg')

"""def get_frequency_band(index, sample_rate, frame_size):
    freq = index * sample_rate / frame_size
    if 0 <= freq < 60:
        return 0 #"Sub-bass"
    elif 60 <= freq < 250:
        return 1 #"Bass"
    elif 250 <= freq < 500:
        return 2 #"Low midrange"
    elif 500 <= freq < 2000:
        return 3 #"Midrange"
    elif 2000 <= freq < 4000:
        return 4 # "Upper midrange"
    elif 4000 <= freq < 6000:
        return 5 # "Presence"
    elif 6000 <= freq :
        return 6 # "Brilliance"
    else:
        print(f"freq {freq} hz")
        return 7 # "Out of range" """
        
def get_frequency_band(index, sample_rate, frame_size):
    freq = index * sample_rate / frame_size
    if 0 <= freq < 500:
        return 0 #"Sub-bass,bass,lowmidrange"
    elif 500 <= freq < 2000:
        return 1 #"Midrange"
    elif 2000 <= freq < 4000:
        return 2 # "Upper midrange"
    elif 4000 <= freq < 6000:
        return 3 # "Presence"
    elif 6000 <= freq :
        return 4 # "Brilliance"
    else:
        print(f"freq {freq} hz")
        return 7 # "Out of range"

def band_to_hue(band):
    #hue_values = [0.1, 0.2, 0.4, 0.6, 0.8]
    hue_values = [269/360, 30/360, 296/360, 339/360, 49/360]
    return hue_values[band]

"""def band_to_hue(band):
    hue_values = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    return hue_values[band]"""


"""def band_to_numbins(band): #based on get_frequency_band values
    if band == 0:
        return 2 #"Sub-bass"
    elif band == 1:
        return 4 #"Bass"
    elif band == 2:
        return 5 #"Low midrange"
    elif band == 3:
        return 32 #"Midrange"
    elif band == 4:
        return 43 # "Upper midrange"
    elif band == 5:
        return 42 # "Presence"
    elif band == 6 :
        return 384 # "Brilliance"
    else:
        print(f"weird band {band}")
        return 0 # "Out of range" """
        
        
def band_to_numbins(band,sample_rate=44100,frame_size=1024): #based on get_frequency_band values
    freq_res = sample_rate / frame_size
    freq_bands = [500,2000,4000,6000,sample_rate/2]
    if band == 0:
        return round((freq_bands[0]-0)/freq_res) #"Sub-bass,bass,lowmidrange"
    elif band == 1:
        return round((freq_bands[1]-freq_bands[0])/freq_res)#32 #"Midrange"
    elif band == 2:
        return round((freq_bands[2]-freq_bands[1])/freq_res)#43 # "Upper midrange"
    elif band == 3:
        return round((freq_bands[3]-freq_bands[2])/freq_res)#42 # "Presence"
    elif band == 4 :
        return int((freq_bands[4]-freq_bands[3])/freq_res)#384 # "Brilliance"
    else:
        print(f"weird band {band}")
        return 0 # "Out of range"

#@profile
def apply_mdct(audio, frame_size, hop_size, sample_rate):
    num_frames = 1 + (len(audio) - frame_size) // hop_size
    #print(f'len: {len(audio)}, numframes: {num_frames}')
    mdct_frames = np.empty((num_frames, frame_size), dtype=np.float32)
    
    # Define the window function
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_size) / frame_size))
    #window = np.sin(np.pi * (np.arange(frame_size) + 0.5) / frame_size)

    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size
        windowed_frame = audio[start:end] * window
        mdct_frames[n] = fftpack.dct(windowed_frame, type=2, norm='ortho')
        #mdct_frames[n] = fftpack.dct(windowed_frame, type=2, norm=None)
    
    mdct_out = mdct_frames[:, :frame_size//2]

    return mdct_out

#@profile
def mdct_to_hsv(mdct_array, frame_size, sample_rate, saturation=1):

    #print(f"mdct min: {min_value}")
    #print(f"mdct max: {max_value}")
    abs_mdct = np.abs(mdct_array)
    log_mdct = np.log10(abs_mdct+1) #for handling of zero
    min_log_value = np.min(log_mdct) #should be 0
    max_log_value = np.max(log_mdct)
    # brightness will be between 0 and 1 for the basolute value of mdct
    brightness = (log_mdct - min_log_value) / (max_log_value - min_log_value)
    
    #print(f"min log: {min_log_value}")
    #print(f"max log: {max_log_value}")
    
    #num_freq_bins = mdct_array.shape[1]
    #freqs = np.fft.fftfreq(frame_size, d=1/sample_rate)[:num_freq_bins]
    
    hue = np.zeros(frame_size//2)   

    print(f"freq res: {sample_rate / frame_size}, max freq: {511*sample_rate / frame_size}")
    for col in range(hue.shape[0]):
        band = get_frequency_band(col, sample_rate, frame_size)
    
        hue[col] = band_to_hue(band)
    
    hue = hue[np.newaxis, :]
    hue = np.repeat(hue, mdct_array.shape[0], axis=0)

    # Find the maximum absolute value in mdct_array
    max_abs_value = np.max(np.abs(mdct_array))

    # Calculate the scaling factor
    scaling_factor = 0.2 / max_abs_value

    # Create the sat_mdct_array with the transformation
    saturation = 0.8 + scaling_factor * mdct_array
    
    hsv_array = np.stack((hue, saturation, brightness), axis=-1)
    #print(f"HSV array shape: {hsv_array.shape}")

    return hsv_array

def calc_req_radius(maxnumcol,dpi=300):
    width_px = 2*8 * (maxnumcol + np.sqrt(2) * maxnumcol)  # Calculate pixel width 

    # Convert pixels to inches
    width_in = width_px / dpi

    return width_in

#@profile
def plot_polar_spectrogram(hsv_array, output_path,dpi=1000,sample_rate=44100,frame_size=1024):
    output_folder = Path(output_path)
    output_filepath = output_folder #/ output_filename
    
    num_rows, num_cols = hsv_array.shape[:2]
    numbands=5
    max_radius = calc_req_radius(band_to_numbins(4,sample_rate,frame_size),dpi)
    print(f'max radius={max_radius}')
    min_radius = max_radius / 8
    annulus_thickness = (max_radius - min_radius) / numbands

    # Convert HSV array to RGB
    rgb_array = hsv_to_rgb(hsv_array)
    del hsv_array

    # Initialize the polar plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Plot annulus for each frequency band
    band_radii = np.linspace(min_radius, max_radius, numbands+1)[::-1]

    current_col = 0
    for band in range(numbands):
        outer_radius = band_radii[band]
        inner_radius = band_radii[band + 1]
        num_band_columns = band_to_numbins(band,sample_rate,frame_size)
        num_band_columns = min(num_band_columns, rgb_array.shape[1] - current_col) #there can be rounding errors which result in more than 512 bands
        # Get the columns corresponding to the current frequency band
        band_columns = list(range(current_col, current_col + num_band_columns))
        #print(f"numband_columns: {num_band_columns}")

        current_col += num_band_columns

        # Create a polar grid 
        theta, r = np.meshgrid(
            np.linspace(0, 2 * np.pi, num_rows+1),
            #np.logspace(np.log2(inner_radius), np.log2(outer_radius), num_band_columns, base=2)[::-1],
            np.linspace(inner_radius, outer_radius, num_band_columns+1),#[:-1],  
            indexing="ij",
        )
        
        print(f"band: {band}, size:{len(band_columns)} or {num_band_columns}, inner radius: {inner_radius}, outer radius: {outer_radius}")
        # Plot the annulus for the current band
        band_rgb_array = rgb_array[:, band_columns, :]
        ax.pcolormesh(theta, r, band_rgb_array, shading="flat")

        if band < 6:
            # Plot the boundary between the current band and the next one
            ax.plot([0, 2 * np.pi], [outer_radius] * 2, 'k', linewidth=1, alpha=0.5)

    ax.set_yticklabels([])  # Remove y-axis ticks
    ax.set_xticklabels([])  # Remove x-axis ticks
    ax.axis("off")  # Remove axis

    # Save the plot as a PNG file
    plt.savefig(output_filepath, transparent=True, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

def audio2png(input_file,output_path,output_file='output.png',start_time=None, end_time=None):
    output_folder = Path(output_path)
    output_filepath = output_folder #/ output_file
    dpi=1000
    frame_size = 1024
    hop_size = frame_size // 2 
    
    audio, sample_rate = librosa.load(input_file, sr=None)
    print(f"samplerate:{sample_rate}")
    # Extract the specified segment of the input audio
    if start_time is not None and end_time is not None:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio = audio[start_sample:end_sample]
             
    mdct_array = apply_mdct(audio, frame_size, hop_size, sample_rate)
    
    #convert mdct frames to HSV array
    hsv_array = mdct_to_hsv(mdct_array,frame_size, sample_rate)
    
    #plot HSV array to polar and output png
    plot_polar_spectrogram(hsv_array,output_path,dpi,sample_rate,frame_size)
    

if __name__ == "__main__":
    audio2png('C:/Users/nnjoo/OneDrive/Documents/polar spectrogram/kids.mp3','C:/Users/nnjoo/OneDrive/Documents/polar spectrogram','kidsspec.png',0,30)
