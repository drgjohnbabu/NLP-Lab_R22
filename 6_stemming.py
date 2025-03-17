# Import necessary libraries
import nltk
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
nltk.download('punkt')

def perform_stemming(text):
    """
    Performs stemming on the input text using the Lancaster Stemmer.
    
    Args:
    text (str): The input text to be stemmed.
    
    Returns:
    list: A list of stemmed words.
    """
    # Initialize Lancaster Stemmer
    stemmer = LancasterStemmer()

    # Tokenize the text into words
    words = word_tokenize(text)

    # Apply stemming to each word
    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words

def main():
    """
    Main function to demonstrate stemming.
    """
    # Sample text for demonstration
    text = 'Focusing is most important for reading writing understanding and for any skill development'

    print("Original Text:")
    print(text)

    # Perform stemming
    stemmed_text = perform_stemming(text)

    # Display results
    print("\nStemmed Text:")
    print(stemmed_text)

if __name__ == "__main__":
    main()