#!/usr/bin/env python

import logging
import os
import uuid
import wave
import json
import tornado.ioloop
import tornado.websocket
import tornado.httpserver
import tornado.web
import webrtcvad
import numpy as np
import librosa
import pickle

from base64 import b64decode
from dotenv import load_dotenv

load_dotenv()

logging.captureWarnings(True)

# Constants:
MS_PER_FRAME = 15  # Duration of a frame in ms

# Load the pre-trained model
loaded_model = pickle.load(open("models/GaussianNB-20190130T1233.pkl", "rb"))
print(loaded_model)

# Global variables
clients = []

class BufferedPipe(object):
    def __init__(self, max_frames, sink):
        self.sink = sink
        self.max_frames = max_frames
        self.count = 0
        self.payload = b''

    def append(self, data):
        self.count += 1
        self.payload += data
        if self.count == self.max_frames:
            self.process()

    def process(self):
        self.sink(self.count, self.payload)
        self.count = 0
        self.payload = b''

class AudioProcessor(object):
    def __init__(self, rate, clip_min):
        self.rate = rate
        self.clip_min_frames = clip_min // MS_PER_FRAME

    def process(self, count, payload):
        if count > self.clip_min_frames:
            fn = f"rec-{uuid.uuid4().hex}.wav"
            with wave.open(fn, 'wb') as output:
                output.setparams((1, 2, self.rate, 0, 'NONE', 'not compressed'))
                output.writeframes(payload)
            logging.debug(f'File written {fn}')
            self.process_file(fn)
            os.remove(fn)
        else:
            logging.info(f'Discarding {count} frames')

    def process_file(self, wav_file):
        if loaded_model is not None:
            X, sample_rate = librosa.load(wav_file, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            X = [mfccs]
            prediction = loaded_model.predict(X)
            logging.info(f"Prediction: {prediction}")

            beep_captured = prediction[0] == 0
            if beep_captured:
                logging.info("Beep detected")

            for client in clients:
                client.write_message({"beep_detected": beep_captured})
        else:
            logging.error("Model not loaded")

class WSHandler(tornado.websocket.WebSocketHandler):
    def initialize(self):
        self.frame_buffer = None
        self.vad = webrtcvad.Vad()
        self.id = uuid.uuid4().hex
        self.rate = None
        self.silence = 20

    def open(self):
        logging.info("Client connected")
        clients.append(self)

    def on_message(self, message):
        if isinstance(message, bytes):
            if self.vad.is_speech(message, self.rate):
                logging.debug("SPEECH detected")
                self.frame_buffer.append(message)
            else:
                logging.debug("Silence detected")
                self.silence -= 1
                if self.silence == 0:
                    self.frame_buffer.process()
        else:
            data = json.loads(message)
            self.rate = 16000
            clip_min = int(data.get('clip_min', 200))
            clip_max = int(data.get('clip_max', 10000))
            silence_time = int(data.get('silence_time', 300))
            sensitivity = int(data.get('sensitivity', 3))

            self.vad.set_mode(sensitivity)
            self.silence = silence_time // MS_PER_FRAME
            self.processor = AudioProcessor(self.rate, clip_min).process
            self.frame_buffer = BufferedPipe(clip_max // MS_PER_FRAME, self.processor)
            self.write_message('ok')

    def on_close(self):
        logging.info("Client disconnected")
        clients.remove(self)

class PingHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('ok')
        self.set_header("Content-Type", 'text/plain')
        self.finish()

def main():
    try:
        logging.basicConfig(level=logging.INFO, format="%(levelname)7s %(message)s")
        application = tornado.web.Application([
            (r"/ping", PingHandler),
            (r"/(.*)", WSHandler),
        ])
        http_server = tornado.httpserver.HTTPServer(application)
        port = int(os.getenv('PORT', 8000))
        http_server.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
