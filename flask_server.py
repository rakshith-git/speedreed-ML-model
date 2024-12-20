from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import spacy
import torch
import logging
from logging.handlers import RotatingFileHandler
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import signal
from simple_model import SimpleRSVPTrainer
from contextlib import contextmanager
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)

@dataclass
class ModelConfig:
    """Model configuration settings"""
    model_path: str = os.getenv('MODEL_PATH', 'best_simple_rsvp_model.pth')
    spacy_model: str = os.getenv('SPACY_MODEL', 'en_core_web_lg')
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_text_length: int = int(os.getenv('MAX_TEXT_LENGTH', 10000))
    batch_size: int = int(os.getenv('BATCH_SIZE', 32))
    request_timeout: int = int(os.getenv('REQUEST_TIMEOUT', 30))
    cache_size: int = int(os.getenv('CACHE_SIZE', 1000))

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

# @contextmanager
# def timeout(seconds: int):
#     """Context manager for timeouts"""
#     def signal_handler(signum, frame):
#         raise TimeoutError("Request timed out")
#
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

# Remove the signal-based context manager and use simple time-based checking instead
def process_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Execute function with timeout"""
    start_time = time.time()
    result = func(*args, **kwargs)
    if time.time() - start_time > timeout_seconds:
        raise TimeoutError("Request timed out")
    return result

class ModelManager:
    """Manages model loading and cleanup"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.config = ModelConfig()
            self.nlp = None
            self.predictor = None
            self.initialized = True
    
    def load_models(self):
        """Lazy load models"""
        if self.nlp is None:
            logger.info("Loading spaCy model...")
            self.nlp = spacy.load(self.config.spacy_model)
        
        if self.predictor is None:
            logger.info("Loading RSVP model...")
            self.predictor = RSVPPredictor(self.config.model_path)
    
    def cleanup(self):
        """Cleanup model resources"""
        if self.predictor:
            self.predictor.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class RSVPPredictor:
    """RSVP prediction handler with caching and resource management"""
    def __init__(self, model_path: str):
        self.trainer = SimpleRSVPTrainer()
        self.trainer.load_model(model_path)
        self.trainer.model.eval()
        self.device = ModelConfig().device
    
    @lru_cache(maxsize=ModelConfig().cache_size)
    def get_delays(self, sentence: str, base_delay: float = 0.2) -> Dict[str, Any]:
        """Get word delays for a sentence with caching"""
        try:
            predictions = process_with_timeout(
                self.trainer.predict,
                ModelConfig().request_timeout,
                sentence,
                base_delay
            )
            return {
                "sentence": sentence,
                "delays": [
                    {"word": word, "delay": float(delay)}
                    for word, delay in predictions
                ]
            }
        except Exception as e:
            logger.error(f"Error processing sentence: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup predictor resources"""
        del self.trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def create_app() -> Flask:
    """Application factory"""
    app = Flask(__name__)
    CORS(app)
    
    # Configure rate limiting
    limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Initialize model manager
    model_manager = ModelManager()
    
    with app.app_context():
        model_manager.load_models()
    
    @app.teardown_appcontext
    def cleanup_models(exception):
        """Cleanup models on app teardown"""
        model_manager.cleanup()
    
    @app.route('/health', methods=['GET'])
    def health_check() -> Tuple[Dict[str, str], int]:
        """Health check endpoint"""
        return jsonify({"status": "healthy"}), 200
    
    @app.route('/metrics', methods=['GET'])
    def metrics() -> Tuple[Dict[str, Any], int]:
        """Basic metrics endpoint"""
        return jsonify({
            "device": model_manager.config.device,
            "cache_info": model_manager.predictor.get_delays.cache_info()._asdict() if model_manager.predictor else {},
            "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else "N/A"
        }), 200
    
    @app.route('/process-text', methods=['POST'])
    @limiter.limit("10 per minute")
    def process_text() -> Tuple[Dict[str, Any], int]:
        """Process text endpoint with validation and batching"""
        start_time = time.time()
        
        # Request validation
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        if not isinstance(text, str):
            return jsonify({"error": "Text must be a string"}), 400
        
        if len(text) > model_manager.config.max_text_length:
            return jsonify({"error": f"Text exceeds maximum length of {model_manager.config.max_text_length}"}), 400
        
        base_delay = data.get('base_delay', 0.25)
        if not isinstance(base_delay, (int, float)) or base_delay <= 0:
            return jsonify({"error": "Invalid base_delay value"}), 400
        
        try:
            # Process text in batches
            doc = model_manager.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            
            results = []
            errors = []
            
            for i in range(0, len(sentences), model_manager.config.batch_size):
                batch = sentences[i:i + model_manager.config.batch_size]
                for sentence in batch:
                    if not sentence:
                        continue
                    try:
                        sentence_delays = model_manager.predictor.get_delays(
                            sentence, base_delay
                        )
                        results.append(sentence_delays)
                    except Exception as e:
                        logger.error(f"Error processing sentence: {str(e)}")
                        errors.append({
                            "sentence": sentence,
                            "error": str(e)
                        })
            
            processing_time = time.time() - start_time
            
            response = {
                "results": results,
                "metadata": {
                    "processed_sentences": len(results),
                    "failed_sentences": len(errors),
                    "processing_time": processing_time,
                }
            }
            
            if errors:
                response["errors"] = errors
            
            return jsonify(response), 200
            
        except TimeoutError:
            return jsonify({"error": "Request timed out"}), 408
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=False
    )
