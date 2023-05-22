import json
from acrcloud.recognizer import ACRCloudRecognizer
from acrcloud.recognizer import ACRCloudRecognizeType
import sys
import os

def parse_result(result):
    data = json.loads(result)
    print('ACRCloud API status:', data['status']['msg'])
    if data['status']['msg'] != 'Success':
        print('No result found')
        return None

    song = data['metadata']['music'][0]
    return song['title'], song['artists'][0]['name']

def identify_song(file_path):
    access_key = os.getenv('ACR_ACCESS_KEY')
    access_secret = os.getenv('ACR_ACCESS_SECRET')

    print(f"First 4 digits of Access Key: {access_key[:4]}")
    print(f"First 4 digits of Access Secret: {access_secret[:4]}")

    config = {
        'host': 'identify-us-west-2.acrcloud.com',
        'access_key': access_key,
        'access_secret': access_secret,
        'recognize_type': ACRCloudRecognizeType.ACR_OPT_REC_AUDIO, 
        'debug': False,
        'timeout': 10  # seconds
    }

    print(f"Currently working with file at: {file_path}")

    recognizer = ACRCloudRecognizer(config)
    buf = open(file_path, 'rb').read()
    print('Buffer size:', len(buf))
    print('Making API call...')
    result = recognizer.recognize_by_filebuffer(buf, 0)
    print('API call complete.')

    song_info = parse_result(result)
    if song_info is not None:
        title, artist = song_info
        print(f'Title: {title}, Artist: {artist}')
        return title, artist
    else:
        print("Could not identify the song.")
        return None, None


def main():
    """if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_audio_file>")
        return

    file_path = sys.argv[1]"""
    file_path = "C:/Users/nnjoo/OneDrive/Documents/polar spectrogram/kidsouttest.wav"
    identify_song(file_path)

if __name__ == "__main__":
    main()