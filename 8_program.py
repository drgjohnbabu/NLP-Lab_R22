
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import Counter

# Download required resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
    """
    Converts POS tag to format compatible with WordNet Lemmatizer.
    """
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if not found

def morphological_analysis(sentence):
    """
    Performs morphological analysis on a sentence using lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return " ".join(lemmatized_words)

def generate_ngrams(text, n):
    """
    Generates n-grams from a given text.
    """
    tokens = word_tokenize(text)
    return list(ngrams(tokens, n))

def laplace_smoothing(ngrams_list):
    """
    Implements Laplace (add-one) smoothing for n-grams.
    """
    ngram_counts = Counter(ngrams_list)
    total_ngrams = sum(ngram_counts.values())
    vocab_size = len(set(ngrams_list))
    smoothed_probs = {ngram: (count + 1) / (total_ngrams + vocab_size) for ngram, count in ngram_counts.items()}
    return smoothed_probs, ngram_counts

def main():
    """
    Main function to demonstrate morphological analysis, n-gram generation, and n-gram smoothing.
    """
    # Morphological Analysis
    sentence = "The running dogs are barking loudly."
    lemmatized_sentence = morphological_analysis(sentence)
    print(f"Original sentence: {sentence}")
    print(f"Morphological analysis (lemmatized): {lemmatized_sentence}")
    
    # N-gram Generation
    text = "The quick brown fox jumps over the lazy dog"
    n = 3  # Trigrams
    trigrams_list = generate_ngrams(text, n)
    print(f"\nGenerated {n}-grams:")
    print(trigrams_list)
    
    
    
    # N-gram Smoothing
    smoothed_probs, _ = laplace_smoothing(trigrams_list)
    print("\nN-gram probabilities after Laplace smoothing:")
    for ngram, prob in smoothed_probs.items():
        print(f"{ngram}: {prob:.4f}")

if __name__ == "__main__":
    main()
