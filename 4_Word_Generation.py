import random
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Download necessary NLTK resources
nltk.download('punkt')

def train_bigram_model(text):
    """
    Trains a simple bigram-based word generation model.

    Args:
    text (str): The input text.

    Returns:
    dict: A dictionary mapping words to their possible next words.
    """
    words = word_tokenize(text.lower())
    bigram_model = defaultdict(list)

    for i in range(len(words) - 1):
        bigram_model[words[i]].append(words[i + 1])
    print(bigram_model)
    return bigram_model

def generate_text(bigram_model, start_word, num_words=10):
    """
    Generates a sequence of words based on a trained bigram model.

    Args:
    bigram_model (dict): Trained bigram word map.
    start_word (str): The word to start generation.
    num_words (int): Number of words to generate.

    Returns:
    str: Generated sentence.
    """
    current_word = start_word.lower()
    generated_words = [current_word]

    for _ in range(num_words - 1):
        if current_word in bigram_model:
            next_word = random.choice(bigram_model[current_word])
            generated_words.append(next_word)
            current_word = next_word
        else:
            break  # Stop if no next word exists
    print(' '.join(generated_words))
    return ' '.join(generated_words)

# Example Usage
if __name__ == "__main__":
    sample_text = """He is a generous human being. 
                 He loves to help others and always shares his knowledge.
                 Generosity and kindness are his true nature.
                 People respect him for his wisdom and humility."""

    model = train_bigram_model(sample_text)
    print("\n start_word: help")
    print("\nGenerated Text:")
    print(generate_text(model, start_word="nature", num_words=15))
