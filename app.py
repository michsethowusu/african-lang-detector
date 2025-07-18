import csv
import os
import re
from typing import Dict, List, Tuple
from pathlib import Path
import math
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigramLanguageModel:
    """Represents bigram patterns for a single language"""

    def __init__(self, language_code: str):
        self.language_code = language_code
        self.bigrams = {}
        self.total_start_count = 0
        self.total_middle_count = 0
        self.total_end_count = 0

    def add_bigram(self, bigram: str, valid_start: bool, valid_middle: bool,
                   valid_end: bool, start_count: int, middle_count: int, end_count: int):
        """Add a bigram pattern to the model"""
        self.bigrams[bigram] = {
            'valid_start': valid_start,
            'valid_middle': valid_middle,
            'valid_end': valid_end,
            'start_count': start_count,
            'middle_count': middle_count,
            'end_count': end_count
        }

        self.total_start_count += start_count
        self.total_middle_count += middle_count
        self.total_end_count += end_count

    def get_bigram_probability(self, bigram: str, position: str) -> float:
        """Get probability of bigram at given position (start, middle, end)"""
        if bigram not in self.bigrams:
            return 0.0

        data = self.bigrams[bigram]

        if position == 'start':
            return data['start_count'] / max(self.total_start_count, 1) if data['valid_start'] else 0.0
        elif position == 'middle':
            return data['middle_count'] / max(self.total_middle_count, 1) if data['valid_middle'] else 0.0
        elif position == 'end':
            return data['end_count'] / max(self.total_end_count, 1) if data['valid_end'] else 0.0

        return 0.0


class AfricanLanguageDetector:
    """Main language detection engine"""

    def __init__(self):
        self.models = {}
        self.smoothing_factor = 1e-6

    def load_language_model(self, csv_file_path: str, language_code: str):
        model = BigramLanguageModel(language_code)

        try:
            with open(csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    bigram = row['bigram']
                    valid_start = row['valid_start'] == 'Y'
                    valid_middle = row['valid_middle'] == 'Y'
                    valid_end = row['valid_end'] == 'Y'
                    start_count = int(row['start_count'])
                    middle_count = int(row['middle_count'])
                    end_count = int(row['end_count'])

                    model.add_bigram(bigram, valid_start, valid_middle, valid_end,
                                     start_count, middle_count, end_count)

            self.models[language_code] = model
            logger.info(f"Loaded {len(model.bigrams)} bigrams for language: {language_code}")

        except Exception as e:
            logger.error(f"Error loading {csv_file_path}: {e}")

    def load_all_models(self, data_directory: str):
        path = Path(data_directory)
        if not path.exists():
            logger.error(f"Data directory not found: {data_directory}")
            return

        for csv_file in path.glob("*.csv"):
            language_code = csv_file.stem
            self.load_language_model(str(csv_file), language_code)

        logger.info(f"Total languages loaded: {len(self.models)}")

    def extract_bigrams(self, text: str) -> List[Tuple[str, str]]:
        text = re.sub(r'[^a-zA-ZɔɛàáèéìíòóùúâêîôûñçÀÁÈÉÌÍÒÓÙÚÂÊÎÔÛÑÇ\s]', '', text)
        text = text.lower().strip()

        bigrams = []
        words = text.split()

        for word in words:
            if len(word) < 2:
                continue

            for i in range(len(word) - 1):
                bigram = word[i:i + 2]
                if i == 0:
                    position = 'start'
                elif i == len(word) - 2:
                    position = 'end'
                else:
                    position = 'middle'
                bigrams.append((bigram, position))

        return bigrams

    def calculate_language_score(self, text: str, language_code: str) -> float:
        """Calculate score for a language using only start and end bigrams"""
        if language_code not in self.models:
            return float('-inf')

        model = self.models[language_code]
        bigrams = self.extract_bigrams(text)

        if not bigrams:
            return float('-inf')

        log_probs = []  # Store log probabilities for start/end bigrams
        position_hits = {'start': False, 'end': False}  # Track if we see at least one start/end

        for bigram, position in bigrams:
            # Skip middle bigrams
            if position == 'middle':
                continue

            prob = model.get_bigram_probability(bigram, position)
            if prob == 0.0:
                prob = self.smoothing_factor
            else:
                position_hits[position] = True

            log_probs.append(math.log(prob))

        # Require at least one start OR end bigram
        if not any(position_hits.values()):
            return float('-inf')

        return sum(log_probs) / len(log_probs)  # Average of start/end log probs

    def detect_language(self, text: str, top_n: int = 5) -> List[Dict]:
        if not self.models:
            return []

        scores = {
            code: self.calculate_language_score(text, code)
            for code in self.models
        }

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for i, (lang_code, score) in enumerate(sorted_scores[:top_n]):
            confidence = min(100.0, max(0.0, (score + 20) * 5))
            results.append({
                'language': lang_code,
                'confidence': round(confidence, 2),
                'score': round(score, 4),
                'rank': i + 1
            })

        return results

    def get_supported_languages(self) -> List[str]:
        return list(self.models.keys())


# Initialize detector globally
detector = AfricanLanguageDetector()

# Try possible model directories
for path in ['./language_data/', './data/', '/app/language_data/', os.path.join(os.path.dirname(__file__), 'language_data')]:
    if os.path.exists(path):
        detector.load_all_models(path)
        break
else:
    logger.warning("No language data directory found.")

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'African Language Detection API',
        'version': '1.0.0',
        'endpoints': {
            'detect': 'POST /detect',
            'languages': 'GET /languages',
            'health': 'GET /health'
        },
        'supported_languages': len(detector.get_supported_languages()),
        'status': 'running'
    })

@app.route('/detect', methods=['POST'])
def detect_language_endpoint():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400

        text = data['text']
        top_n = data.get('top_n', 5)

        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        results = detector.detect_language(text, top_n)

        return jsonify({
            'success': True,
            'text': text,
            'detected_languages': results,
            'total_languages_checked': len(detector.get_supported_languages())
        })

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'supported_languages': detector.get_supported_languages(),
        'total_count': len(detector.get_supported_languages())
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(detector.models),
        'memory_usage': 'OK',
        'service': 'African Language Detection API'
    })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting server on port {port}")
    logger.info(f"Models loaded: {len(detector.models)}")
    app.run(host='0.0.0.0', port=port, debug=debug)
