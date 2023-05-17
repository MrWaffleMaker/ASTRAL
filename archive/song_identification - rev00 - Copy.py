import json
from acrcloud.recognizer import ACRCloudRecognizer
from acrcloud.recognizer import ACRCloudRecognizeType
import sys

def parse_result(result):
    data = json.loads(result)

    if data['status']['msg'] != 'Success':
        print('No result found')
        return None

    song = data['metadata']['music'][0]
    return song['title'], song['artists'][0]['name']

def identify_song(file_path):
    config = {
        'host': 'identify-us-west-2.acrcloud.com',
        'access_key': '1cc0d5d5a0fc3f27c1857f68475859f0',
        'access_secret': 'p5bk9W39mUT2n1uQLh1fyW1ZPc7eAWxtG9JpYZJ4',
        'recognize_type': ACRCloudRecognizeType.ACR_OPT_REC_AUDIO, 
        'debug': False,
        'timeout': 10  # seconds
    }

    recognizer = ACRCloudRecognizer(config)
    buf = open(file_path, 'rb').read()
    result = recognizer.recognize_by_filebuffer(buf, 0)

    song_info = parse_result(result)
    if song_info is not None:
        title, artist = song_info
        print(f'Title: {title}, Artist: {artist}')
    else:
        print("Could not identify the song.")


def main():
    """if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_audio_file>")
        return

    file_path = sys.argv[1]"""
    file_path = "pizzaout.wav"
    identify_song(file_path)

if __name__ == "__main__":
    main()