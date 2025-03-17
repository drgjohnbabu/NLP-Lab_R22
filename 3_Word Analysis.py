import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# Download necessary NLTK resources
nltk.download('punkt')

def word_analysis(text):
    """
    Analyzes the given text to provide frequency distribution and basic statistics.

    Args:
    text (str): The input text.

    Returns:
    None
    """
    # Tokenization
    words = word_tokenize(text.lower())  # Convert to lowercase for uniformity
    words = [word for word in words if word.isalpha()]  # Remove punctuation
    print(words)
    # Frequency Analysis
    word_freq = Counter(words)
    
    # Basic Statistics
    unique_words = len(set(words))
    print(f"len(words {len(words)}")
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    # Display Results
    print("\nWord Frequency Distribution:")
    for word, freq in word_freq.most_common(10):  # Display top 10 words
        print(f"{word}: {freq}")

    print("\nBasic Text Statistics:")
    print(f"Total Words: {len(words)}")
    print(f"Unique Words: {unique_words}")
    print(f"Average Word Length: {avg_word_length:.2f}")

# Example Usage
if __name__ == "__main__":
    sample_text = """Natural Language Processing is an exciting field of Artificial Intelligence.
    It helps computers understand, interpret, and generate human language efficiently."""
    
    word_analysis(sample_text)
