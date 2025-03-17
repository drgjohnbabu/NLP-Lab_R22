import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

# Download necessary resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Sample list of ambiguous words and sentences
sample_sentences = {
    "bank": "I went for a walk along the side of the bank.",
    "bat": "Schin Tendulkar used a very heavy bat during his innings.",
    "crane": "The crane lifted heavy steel beams at the construction site.",
    "date": "Our first date was at a coffee shop.",
    "light1": "This bag is very light to carry.",  # Light (Not heavy)
    "light2": "it was a light hearted comedy movie.",  # Light (Illumination)
    "spring": "The flowers bloom in spring.",
    "bass": "He caught a huge bass while fishing.",
    "rock": "The rock is difficult to break.",
    "watch": "apple watch is very costly."
}

def perform_wsd(word, sentence):
    """
    Implements Word Sense Disambiguation (WSD) using the Lesk algorithm.
    
    Args:
    word (str): The ambiguous word for which WSD is performed.
    sentence (str): The sentence containing the ambiguous word.
    
    Returns:
    str: The best-matching sense of the word based on context.
    """
    tokens = word_tokenize(sentence)  # Tokenize the sentence
    actual_word = word.replace("1", "").replace("2", "")  # Remove numbering from "light1", "light2"
    sense = lesk(tokens, actual_word)  # Apply the Lesk algorithm
    
    return sense.definition() if sense else "No suitable sense found"

# Run WSD on all sample words
for word, sentence in sample_sentences.items():
    sense = perform_wsd(word, sentence)
    print(f"Word: {word.replace('1', '').replace('2', '')}")  # Remove numbering from print output
    print(f"Sentence: {sentence}")
    print(f"Predicted Sense: {sense}\n")
