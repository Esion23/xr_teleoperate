import os
import time
import json
import threading
import numpy as np

class IndependentErrorLogger:
    def __init__(self, log_dir="./utils/data/error_logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"error_log_{timestamp}.jsonl")
        self.file = open(self.log_file, 'w')
        self.lock = threading.Lock()
        
    def log(self, timestamp, errors):
        """
        Log error data to file.
        errors: dict containing error metrics
        """
        data = {
            "timestamp": timestamp,
            **errors
        }
        
        with self.lock:
            self.file.write(json.dumps(data) + "\n")
            self.file.flush() # Ensure data is written immediately
            
    def close(self):
        with self.lock:
            if self.file:
                self.file.close()
