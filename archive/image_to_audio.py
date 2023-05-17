def get_frequency_band(index, sample_rate, frame_size):
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
        return 7 # "Out of range"

def band_to_hue(band):
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
        return 0 # "Out of range"
    
def apply_imdct(mdct_frames, frame_size, hop_size, sample_rate):
    num_frames, num_bins = mdct_frames.shape
    """
    # Calculate min and max frequencies
    freqs = np.fft.fftfreq(frame_size, d=1/sample_rate)[:frame_size//2]
    min_freq, max_freq = freqs.min(), freqs.max()
    print(f"Min frequency: {min_freq} Hz")
    print(f"Max frequency fft: {max_freq} Hz")
    """
    mdct_frames_full = np.zeros((num_frames, frame_size), dtype=np.float32)
    mdct_frames_full[:, :num_bins] = mdct_frames
    mdct_frames_full[:, -num_bins:] = mdct_frames[:, ::-1]

    audio_reconstructed = np.zeros(num_frames * hop_size + frame_size, dtype=np.float32)

    for n in range(num_frames):
        start = n * hop_size
        end = start + frame_size
        windowed_frame = fftpack.idct(mdct_frames_full[n], type=2, norm='ortho') * np.sin(np.pi * (np.arange(frame_size) + 0.5) / frame_size)
        audio_reconstructed[start:end] += windowed_frame * np.sin(np.pi * (np.arange(frame_size) + 0.5) / frame_size)

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

def polar_image_to_hsv(file_name, sample_rate, audio_length, hop_size, frame_size=1024):
    
    
    # Load the input image
    rgb_image = Image.open(file_name).convert('RGB')
   
    # Calculate the center of the image, assuming it is square
    #center = rgb_image.width // 2

    # Calculate the maximum and minimum plot radii
    #max_radius = num_freq_bins
    max_radius = find_circle_radius(file_name)
    print(f"max radius: {max_radius}")
    #max_radius = 10  # inches
    min_radius = max_radius / 8
    band_radii = np.linspace(min_radius, max_radius, 8)[::-1]

    # Initialize the output HSV array
    num_time_bins = 1 + (audio_length - frame_size) // hop_size
    num_cols = frame_size // 2
    num_rows = num_time_bins

    # Calculate the total number of columns in all bands
    total_columns = sum(band_to_numbins(band) for band in range(7))

    # Create the output_hsv array with the correct dimensions
    output_hsv = np.zeros((num_rows, total_columns, 3))
    image_height = rgb_image.height
    image_width = rgb_image.width
    center_x = rgb_image.width // 2
    center_y = rgb_image.height // 2
    print(f"img height: {image_height}, width: {image_width}")
    
    debug_image = Image.new('RGB', (image_width, image_height), "black")
    # Extract the HSV values for each band
    current_column = 0
    for band in range(7):
        outer_radius = band_radii[band]
        inner_radius = band_radii[band + 1]
        band_thickness = outer_radius - inner_radius
        

        # Determine the number of columns in the current band
        num_band_columns = band_to_numbins(band)
        subband_thickness = band_thickness/num_band_columns
        print(f"band: {band}, size:{num_band_columns}, inner radius: {inner_radius}, outer radius: {outer_radius}, band thick: {band_thickness}, subband thickness: {subband_thickness}")
        # Create a polar grid

        theta, r = np.meshgrid(
            np.linspace(0, 2*np.pi, num_rows),
            np.linspace(inner_radius + subband_thickness/2, outer_radius-subband_thickness/2, num_band_columns),  # No need to exclude the last point anymore
            indexing="ij",
        )
        # Compute Cartesian coordinates for each point in the polar grid
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        
        # Convert Cartesian coordinates to image coordinates

        x_coords = (x+image_width/2).astype(int)
        y_coords = (-y+image_height/2).astype(int)

        for col in range(num_band_columns):
            for row in range(num_rows):
                r, g, b = rgb_image.getpixel((x_coords[row, col], y_coords[row, col]))
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                output_hsv[row, current_column + col, :] = [h, s, v]
                debug_image.putpixel((x_coords[row, col], y_coords[row, col]), (255, 255, 255))

                if col == 0 and row == 0:
                    print(f"First r and theta read: x_coord={x_coords[row, col]}, y_coord={y_coords[row, col]}")
        current_column += num_band_columns

    print(f"hsv from png array shape: {output_hsv.shape}")
     
    """
    # Save the value part of the output_hsv array to a CSV file
    with open("output_value.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in output_hsv[:, :, 2]:
            csv_writer.writerow(row)
            """
    debug_image.save("debug_image.png")

    return output_hsv


def find_circle_radius(image_path):
    img = Image.open(image_path).convert('RGBA')
    data = np.array(img)

    # Find the center of the image
    center_x = data.shape[1] // 2
    center_y = data.shape[0] // 2

    # Create an array with the squared distance to the center for each pixel
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    dist_from_center_squared = (x - center_x)**2 + (y - center_y)**2

    # Find the squared radius of the circle: the largest distance for which the alpha channel is not zero
    squared_radius = dist_from_center_squared[data[..., 3] > 0].max()

    # Return the radius. We use np.sqrt because we computed the squared radius.
    return int(np.sqrt(squared_radius))

def main(input_file):
    
    output_file = "pizzaout.wav"
   
    frame_size = 1024
    hop_size = frame_size // 2 
    
    audio, sample_rate = librosa.load(input_file, sr=None)
    print(f"sample_rate: {sample_rate}")
    # Extract the specified segment of the input audio
    if start_time is not None and end_time is not None:
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio = audio[start_sample:end_sample]
        
     # Parameters for converting the polar PNG image back to an HSV array
     
    #duration = end_time-start_time
    duration = len(audio)
    cutoff_pct = 0
    mdct_array = apply_mdct(audio, frame_size, hop_size, sample_rate)

    # Find the minimum and maximum values in the original MDCT array
    min_value = np.min(mdct_array)
    max_value = np.max(mdct_array)
    
    #convert mdct frames to HSV array
    hsv_array = mdct_to_hsv(mdct_array,frame_size, sample_rate)
    #hsv_array = create_dummy_hsv_array(2811, 512)
    
    # Save the HSV array as a polar spectrogram PNG file
    polar_spectrogram_file_name = "polar_spectrogram.png"
    #plot HSV array to polar and output png
    plot_polar_spectrogram(hsv_array,frame_size,sample_rate,polar_spectrogram_file_name)
    
   
    
    # Load the saved polar spectrogram and convert it to an HSV array
    loaded_hsv_array = polar_image_to_hsv(polar_spectrogram_file_name, sample_rate, duration, hop_size, frame_size)
    
    # Convert the loaded HSV array back to an MDCT array
    recovered_mdct_array = hsv_to_mdct(loaded_hsv_array)
    #recovered_mdct_array = hsv_to_mdct(hsv_array)
    
    # Create the polar spectrogram
    #rgb_array = create_polar_spectrogram(mdct_frames, min_freq=0, max_freq=10000, duration=2*np.pi, log_offset=1)
    audio_reconstructed = apply_imdct(recovered_mdct_array, frame_size, hop_size, sample_rate)
    #audio_reconstructed = apply_imdct(mdct_array, frame_size, hop_size, sample_rate)
    
    #save_audio_file(audio, output_file, sample_rate)
    save_audio_file(audio_reconstructed, output_file, sample_rate)

if __name__ == "__main__":
    main()
