suspicious_bigrams = [
    'po box', '1000 cash', 'prize guaranteed', 'send stop',
    'selected receive', 'await collection', 'urgent mobile',
    'land line', 'valid 12hrs', 'customer service'
]
keywords = ['win', 'free', 'urgent', 'prize', 'cash', 'call', 'txt']
question_words = ['what', 'when', 'why', 'how', 'who', 'where']
MAX_FEATURE_TF_IDF = 1000
SELECTED_FEATURES = [
    'num_words',
    'has_keyword',
    'count_suspicious_bigrams',
    'num_uppercase',
    'num_exclamations',
    'num_digits',
    'has_question_word',
    'has_question_mark',
    'ends_with_stop'
]

MAX_DEPTH_DT = 6
MIN_SAMPLES_SPLIT_DT = 20
LOG_DIR = "logs"
IMAGES_DIR = "images"
MODELS_PATH = "models"