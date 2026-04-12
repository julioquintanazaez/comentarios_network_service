import re
from collections import Counter
from typing import List, Set

# Stop words en español
SPANISH_STOP_WORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con",
    "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí",
    "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay",
    "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra",
    "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos",
    "otro", "otras", "otra", "él", "ella", "ellos", "nosotros", "vosotros", "ustedes",
    "mi", "tu", "su", "nuestro", "vuestro", "mis", "tus", "sus", "nuestros", "vuestros",
    "este", "esta", "estos", "estas", "aquel", "aquella", "aquellos", "aquellas",
    "ser", "estar", "tener", "hacer", "poder", "decir", "ir", "ver", "dar", "saber", "querer",
    "llegar", "pasar", "deber", "poner", "parecer", "quedar", "creer", "hablar", "llevar",
    "yo", "tú", "él", "ella", "ello", "nosotros", "vosotros", "ellos", "ellas",
    "me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las"
}

# Stop words en inglés (comunes)
ENGLISH_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
    "its", "of", "on", "that", "the", "to", "was", "were", "will", "with", "i", "you", "we", "they",
    "this", "that", "these", "those", "am", "do", "does", "did", "doing", "have", "having", "can",
    "could", "would", "should", "might", "must", "my", "your", "his", "her", "its", "our", "their",
    "what", "which", "who", "whom", "whose", "why", "how", "then", "than", "so", "too", "very",
    "just", "but", "not", "now", "then", "there", "their", "they're", "there's", "were", "we've",
    "you've", "they've", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "won't", "wouldn't", "shouldn't", "couldn't", "mightn't", "mustn't",
    "like", "just", "some", "any", "no", "only", "own", "same", "than", "then", "these", "those",
    "through", "until", "up", "down", "off", "over", "under", "again", "further", "once", "here",
    "there", "all", "both", "each", "few", "more", "most", "other", "some", "such", "above", "below",
    "between", "during", "without", "within", "along", "across", "behind", "below", "beneath",
    "beside", "beyond", "circa", "except", "including", "plus", "since", "versus", "via"
}

# Combinar ambas stop words
STOP_WORDS = SPANISH_STOP_WORDS.union(ENGLISH_STOP_WORDS)

# Compilar patrones regex una sola vez para mejor rendimiento
PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
NUMBER_PATTERN = re.compile(r'\d+')
MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#(\w+)')
ACCENT_MAP = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'
}
ACCENT_PATTERN = re.compile('|'.join(ACCENT_MAP.keys()))

# Patrón para emojis (compilado una sola vez)
EMOJI_PATTERN = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)

def replace_accent(match):
    """Reemplaza una letra acentuada por su versión sin acento"""
    return ACCENT_MAP.get(match.group(0), match.group(0))

def clean_text(text: str) -> List[str]:
    """
    Limpieza completa de texto:
    - URLs
    - Menciones (@usuario)
    - Hashtags (convierte #ejemplo a ejemplo)
    - Emojis
    - Puntuación
    - Números
    - Acentos
    - Stop words (español e inglés)
    - Palabras de menos de 3 letras
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = URL_PATTERN.sub('', text)
    
    # Eliminar menciones (@usuario)
    text = MENTION_PATTERN.sub('', text)
    
    # Convertir hashtags a palabras (conservar el texto sin #)
    text = HASHTAG_PATTERN.sub(r'\1', text)
    
    # Eliminar emojis
    text = EMOJI_PATTERN.sub('', text)
    
    # Eliminar puntuación y números
    text = PUNCTUATION_PATTERN.sub(' ', text)
    text = NUMBER_PATTERN.sub('', text)
    
    # Eliminar acentos
    text = ACCENT_PATTERN.sub(replace_accent, text)
    
    # Eliminar espacios múltiples y trim
    text = MULTIPLE_SPACES_PATTERN.sub(' ', text).strip()
    
    # Tokenizar y filtrar
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    
    return tokens

def get_word_set(comment_text: str) -> Set[str]:
    """Devuelve set de palabras únicas del comentario"""
    return set(clean_text(comment_text))

def get_word_counts(comment_text: str) -> Counter:
    """Devuelve Counter con frecuencias de palabras"""
    return Counter(clean_text(comment_text))

def show_stop_words_stats() -> dict:
    """Muestra estadísticas de las stop words cargadas"""
    return {
        "total_stop_words": len(STOP_WORDS),
        "spanish_count": len(SPANISH_STOP_WORDS),
        "english_count": len(ENGLISH_STOP_WORDS),
        "sample_spanish": list(SPANISH_STOP_WORDS)[:15],
        "sample_english": list(ENGLISH_STOP_WORDS)[:15]
    }