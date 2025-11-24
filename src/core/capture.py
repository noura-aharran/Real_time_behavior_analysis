# src/core/capture.py
import threading
import cv2
import time

class VideoCaptureAsync:
    """
    Capture vidéo dans un thread séparé pour ne pas bloquer l'UI.
    Usage:
        cap = VideoCaptureAsync(source)
        cap.start()
        ret, frame = cap.read()
        cap.stop()
    """
    def __init__(self, src=0, width=None, height=None, fps=30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.running = False
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

    def _open(self):
        if isinstance(self.src, str) and self.src.isdigit():
            src = int(self.src)
        else:
            src = self.src
        self._cap = cv2.VideoCapture(src)
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, int(self.fps))

    def start(self):
        if self.running:
            return
        self._open()
        self.running = True
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _update(self):
        while self.running and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                # small sleep to avoid busy-loop if stream dies
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = frame
        # release at end
        try:
            self._cap.release()
        except Exception:
            pass

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            if self._cap and self._cap.isOpened():
                self._cap.release()
        except Exception:
            pass
