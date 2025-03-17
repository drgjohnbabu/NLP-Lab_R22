import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required resources if not available
nltk.download('punkt')

def porter_stemming(text):
    """
    Applies Porter Stemming Algorithm to each word in the input text.

    Args:
    text (str): Input sentence or paragraph.

    Returns:
    list: List of words after applying stemming.
    """
    # Initialize Porter Stemmer
    stemmer = PorterStemmer()

    # Tokenize the input text
    words = word_tokenize(text)

    # Apply stemming to each token
    stemmed_words = [stemmer.stem(word) for word in words]

    return stemmed_words

def main():
    # Example text
    text = "Running jogging skipping  excercising workouts and dieting are good for health."

    # Apply stemming
    stemmed_output = porter_stemming(text)

    # Display original and stemmed text
    print("Original Text:")
    print(text)
    print("\nStemmed Output:")
    print(" ".join(stemmed_output))

if __name__ == "__main__":
    main()
