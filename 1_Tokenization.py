import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download NLTK resources if not already downloaded
nltk.download('punkt')

def perform_tokenization(text):
    """
    Performs sentence and word tokenization on the input text.

    Args:
    text (str): The input text to be tokenized.

    Returns:
    tuple: A tuple containing a list of sentences and a list of words.
    """
    # Sentence tokenization
    sentences = sent_tokenize(text)
    
    # Word tokenization
    words = word_tokenize(text)
    
    return sentences, words

def display_tokenization_results(sentences, words):
    """
    Displays the results of sentence and word tokenization.

    Args:
    sentences (list): List of tokenized sentences.
    words (list): List of tokenized words.
    """
    print("\nSentence Tokenization:")
    for i, sentence in enumerate(sentences, start=1):
        print(f"Sentence {i}: {sentence}")
    
    print("\nWord Tokenization:")
    print(words)

def main():
    """
    Main function to demonstrate tokenization.
    """
    # Sample text for demonstration
    text = """Natural Language Processing is an amazing field of Artificial Intelligence. 
    It enables machines to understand human language effectively and accurately."""
    
    print("Original Text:")
    print(text)
    
    # Perform tokenization
    sentences, words = perform_tokenization(text)
    
    # Display tokenization results
    display_tokenization_results(sentences, words)

if __name__ == "__main__":
    main()

