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

"""def band_to_hue(band):
    hue_values = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    return hue_values[band]

def band_to_numbins(band): #based on get_frequency_band values
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
    hue_values = [0.1, 0.2, 0.4, 0.6, 0.8]
    return hue_values[band]

def band_to_numbins(band,sample_rate=48000,frame_size=1024): #based on get_frequency_band values
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
        return round((freq_bands[4]-freq_bands[3])/freq_res)#384 # "Brilliance"
    else:
        print(f"weird band {band}")
        return 0 # "Out of range"


def apply_imdct(mdct_frames, frame_size, hop_size, sample_rate):
    num_frames, num_bins = mdct_frames.shape

    mdct_frames_full = np.zeros((num_frames, frame_size), dtype=np.float32)
    mdct_frames_full[:, :num_bins] = mdct_frames
    mdct_frames_full[:, -num_bins:] = mdct_frames[:, ::-1]

    # Initialize the reconstructed audio signal with zeros
    audio_reconstructed = np.zeros(num_frames * hop_size + frame_size, dtype=np.float32)

    # Define the window function
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_size) / frame_size))

    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size

        # Apply the inverse MDCT and window
        windowed_frame = fftpack.idct(mdct_frames_full[n], type=2, norm='ortho') * window
        #windowed_frame = (fftpack.idct(mdct_frames_full[n], type=2, norm=None) * window) / (frame_size // 2)

        # Add the windowed frame to the audio signal
        audio_reconstructed[start:end] += windowed_frame

    return audio_reconstructed

def hsv_to_mdct(hsv_array):
    hue, saturation, brightness = np.split(hsv_array, 3, axis=-1)
    saturation = saturation.squeeze()
    brightness = brightness.squeeze()
    
    # Inverse transformation for brightness
    min_log_value = np.min(brightness)
    max_log_value = np.max(brightness)
    log_mdct = brightness * (max_log_value - min_log_value) + min_log_value
    abs_mdct = np.power(10, log_mdct) - 1
    
    # Inverse transformation for saturation
    # Estimating the scaling factor by assuming the maximum saturation value corresponds to the max_abs_value
    max_saturation_value = np.max(saturation)
    scaling_factor = (max_saturation_value - 0.8) / 0.2
    if scaling_factor == 0:
        print("0 scaling")
    mdct_approx = (saturation - 0.8) / scaling_factor
    
    # Apply the sign of the approximated MDCT values
    mdct_approx_signed = np.sign(mdct_approx) * abs_mdct

    return mdct_approx_signed

def polar_image_to_hsv(file_path, duration, hop_size, frame_size=1024, sample_rate=44100):
    numbands=5
    audio_length = duration*sample_rate
    #max_radius = num_freq_bins
    max_radius = find_circle_radius(file_path)
    print(f"max dim: {max_radius}")
    # Load the input image
    image = Image.open(file_path).convert('RGBA')

    # Create a new image with the same size as the original image and fill it with black
    black_background = Image.new('RGBA', image.size, (0, 0, 0, 255))

    # Composite the original image onto the black background
    rgb_image = Image.alpha_composite(black_background, image).convert('RGB')
    
    #max_radius = 10  # inches
    min_radius = max_radius / 8
    band_radii = np.linspace(min_radius, max_radius, numbands+1)[::-1]

    # Initialize the output HSV array
    num_time_bins = 1 + (audio_length - frame_size) // hop_size
    num_cols = frame_size // 2
    num_rows = num_time_bins

    # Calculate the total number of columns in all bands
    total_columns = sum(band_to_numbins(band,sample_rate,frame_size) for band in range(numbands))
    
    # Create the output_hsv array with the correct dimensions
    output_hsv = np.zeros((num_rows, total_columns, 3))
    image_height = rgb_image.height
    image_width = rgb_image.width
    center_x = rgb_image.width // 2
    center_y = rgb_image.height // 2
    print(f"image widt: {image_width}")
    debug_image = Image.new('RGB', (image_width, image_height), "black")
    # Extract the HSV values for each band
    current_column = 0
    for band in range(numbands):
        outer_radius = band_radii[band]
        inner_radius = band_radii[band + 1]
        band_thickness = outer_radius - inner_radius
        

        # Determine the number of columns in the current band
        num_band_columns = band_to_numbins(band,sample_rate,frame_size)
        subband_thickness = band_thickness/num_band_columns
        print(f"band: {band}, size:{num_band_columns}, inner radius: {inner_radius}, outer radius: {outer_radius}, band thick: {band_thickness}, numband thickness: {band_to_numbins(band)}")
        # Create a polar grid

        theta, r = np.meshgrid(
            np.linspace(0, 2*np.pi, num_rows),
            np.linspace(inner_radius + subband_thickness/2, outer_radius-subband_thickness/2, num_band_columns),
            indexing="ij",
        )
        # Compute Cartesian coordinates for each point in the polar grid
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        
        # Convert Cartesian coordinates to image coordinates

        x_coords = np.round((x+image_width/2)).astype(int)
        y_coords = np.round((-y+image_height/2)).astype(int)-1 #got this 1 from debugging, maybe theres a 0vs1 indexing diff

        for col in range(num_band_columns):
            for row in range(num_rows):
                r, g, b = rgb_image.getpixel((x_coords[row, col], y_coords[row, col]))
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                output_hsv[row, current_column + col, :] = [h, s, v]
                #debug_image.putpixel((x_coords[row, col], y_coords[row, col]), (255, 255, 255))
                if v == 1:  # If the brightness (value) is 1
                    x = x_coords[row, col]
                    y = y_coords[row, col]
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)  # Euclidean distance
                    #print(f"Pixel with brightness of 1 found at band={band}, col/numbandcol={col/num_band_columns} distance from center={distance}")
                    #print(f"Pixel with brightness of 1 found at x={x_coords[row, col]}, y={y_coords[row, col]}")
                #if col == 0 and row == 0:
                    #print(f"First r and theta read: x_coord={x_coords[row, col]}, y_coord={y_coords[row, col]}")
        current_column += num_band_columns

    debug_image.save("C:/Users/nnjoo/OneDrive/Documents/polar spectrogram/debug_image.png")
    """with open("output_value.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in output_hsv[:, :, 2]:
            csv_writer.writerow(row)"""
    
    return output_hsv
    
def find_circle_radius(image_path):
    img = Image.open(image_path).convert('RGBA')
    data = np.array(img)

    # Find the center of the image
    center_x = data.shape[1] // 2
    center_y = data.shape[0] // 2
    print(f"find circle dim: {center_x}")
    # Create an array with the squared distance to the center for each pixel
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    dist_from_center_squared = (x - center_x)**2 + (y - center_y)**2

    # Find the squared radius of the circle: the largest distance for which the alpha channel is not zero
    squared_radius = dist_from_center_squared[data[..., 3] > 0].max()

    # Return the radius. We use np.sqrt because we computed the squared radius.
    return int(np.sqrt(squared_radius))

def save_audio_file(audio_data, output_path, sample_rate):
    sf.write(output_path, audio_data, sample_rate)

def png2audio(input_file,output_path,output_file='output.png',duration=30, sample_rate=44100,frame_size=1024):
    print(f"Currently working with file at: {input_file}")

    output_folder = Path(output_path)
    output_filepath = output_folder #/ output_file
    
    hop_size = frame_size // 2 

    # Load the saved polar spectrogram and convert it to an HSV array
    loaded_hsv_array = polar_image_to_hsv(input_file, duration, hop_size, frame_size)
    
    # Convert the loaded HSV array back to an MDCT array
    recovered_mdct_array = hsv_to_mdct(loaded_hsv_array)
    #recovered_mdct_array = hsv_to_mdct(hsv_array)
    
    # Create the polar spectrogram
    audio_reconstructed = apply_imdct(recovered_mdct_array, frame_size, hop_size, sample_rate)
    #audio_reconstructed = apply_imdct(mdct_array, frame_size, hop_size, sample_rate)
    
    #save_audio_file(audio, output_file, sample_rate)
    save_audio_file(audio_reconstructed, output_filepath, sample_rate)

if __name__ == "__main__":
    png2audio('C:/Users/nnjoo/OneDrive/Documents/polar spectrogram/kidsspec.png','C:/Users/nnjoo/OneDrive/Documents/polar spectrogram','kidsouttest.wav',30,44100)
