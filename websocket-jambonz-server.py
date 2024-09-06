import logging
import os
import sys
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

log_level_mapping = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

log_level = os.getenv('LOG_LEVEL', 'INFO')
log_level = log_level_mapping.get(log_level.upper(), logging.INFO)

logger = logging.getLogger('freeswitch_audio_stream_poc')
logging.basicConfig(level=log_level)

log_console_handler = logging.StreamHandler(sys.stdout)
log_console_handler.setLevel(log_level)
log_console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(log_console_handler)

logging.captureWarnings(True)

# Constants:
MS_PER_FRAME = 10  # Duration of a frame in ms

# Load the pre-trained model
loaded_model = pickle.load(open("models/rf.pkl", "rb"))
print(loaded_model)

# Global variables
clients = []

# Bearer Token
BEARER_TOKEN = os.getenv('BEARER_TOKEN', 'wby3HSdabFKufpFKTsRsPenaBu7aRt3U96y')


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


def process_file(wav_file):
    if loaded_model is not None:
        x, sample_rate = librosa.load(wav_file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
        x = [mfccs]
        prediction = loaded_model.predict(x)
        logger.info(f"Prediction: {prediction}")

        beep_captured = prediction[0] == 0
        if beep_captured:
            logger.info("Beep detected")

            response = {
                "type": "beep_detected",
            }

            for client in clients:
                client.write_message(json.dumps(response))
    else:
        logger.error("Model not loaded")


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
            logger.debug(f'File written {fn}')
            process_file(fn)
            os.remove(fn)
        else:
            logger.info(f'Discarding {count} frames')


class WSHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, application, request):
        super().__init__(application, request)
        self.processor = None
        self.silence = 20
        self.rate = None
        self.id = uuid.uuid4().hex
        self.vad = webrtcvad.Vad()
        self.frame_buffer = None
        self.path = None
        self.tick = None

    def initialize(self):
        self.frame_buffer = None
        self.vad = webrtcvad.Vad()
        self.id = uuid.uuid4().hex
        self.rate = None
        self.silence = 20

    def check_origin(self, origin):
        return True

    def authenticate(self):
        auth_header = self.request.headers.get('Authorization', None)
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            if token == BEARER_TOKEN:
                return True
        return False

    def open(self, path):
        if not self.authenticate():
            logger.warning("Authentication failed: Invalid or missing token")
            self.close(reason="Authentication failed")
            return

        logger.info("Client connected")
        clients.append(self)
        self.path = self.request.uri
        self.tick = 0

    def on_message(self, message):
        if isinstance(message, bytes):
            try:
                if self.vad.is_speech(message, self.rate):
                    logger.debug("SPEECH detected")
                    self.frame_buffer.append(message)
                else:
                    logger.debug("Silence detected")
                    self.silence -= 1
                    if self.silence == 0:
                        self.frame_buffer.process()
            except Exception as e:
                logger.error(f"Error while parsing message: {str(e)}")
                pass
        else:
            data = json.loads(message)

            if data.get('type') == 'start':
                logger.info("Received 'start' message")
                self.rate = data.get('sampleRateHz', 8000)
                clip_min = int(data.get('clip_min', 200))
                clip_max = int(data.get('clip_max', 10000))
                silence_time = int(data.get('silence_time', 300))
                sensitivity = int(data.get('sensitivity', 0)) # TODO: should probably be set to 3 (most sensitive) in production

                self.vad.set_mode(sensitivity)
                self.silence = silence_time // MS_PER_FRAME
                self.processor = AudioProcessor(self.rate, clip_min).process
                self.frame_buffer = BufferedPipe(clip_max // MS_PER_FRAME, self.processor)
                self.write_message('ok')

            elif data.get('type') == 'stop':
                logger.info("Received 'stop' message")
                if self.frame_buffer:
                    self.frame_buffer.process()
                self.write_message('stopped')
                self.close()

            else:
                logger.warning(f"Received unknown message type: {data.get('type')}")

    def on_close(self):
        logger.info("Client disconnected")
        clients.remove(self)


class PingHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('ok')
        self.set_header("Content-Type", 'text/plain')
        self.finish()


def main():
    try:
        application = tornado.web.Application([
            (r"/ping", PingHandler),
            (r"/(.*)", WSHandler),
        ])
        http_server = tornado.httpserver.HTTPServer(application)
        port = int(os.getenv('PORT', 8000))
        logger.info(f"WS server started on port {port}")
        http_server.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
