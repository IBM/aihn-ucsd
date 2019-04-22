import numpy as np
import collections
from nltk.corpus import stopwords
from utils.text_processing import word_tokenize
from utils.text_processing import sent_tokenize
from utils.text_processing import stem_and_lemmatize
from utils.text_processing import clean_text
from utils.text_processing import tokens_to_ids
from utils.text_processing import normalize_sentence_length

stop_words = set(stopwords.words('english'))

def text_to_batch(text, vocab, maxlen=20, precleaned=False):
    '''
    Turn a sentence into tokens based on vocab and lengths

    Args:
        text (String):  A sentence to convert
        vocab (dict):   A dictionary mapping tokens to ids
        maxlen (int):   The token length to normalize the sentence to
    '''

    if precleaned:
        toks = text.split(' ')
    else:
        # Remove stop words
        nostops = " ".join([t for t in word_tokenize(text) if t not in stop_words])
        # Clean and tokenize
        toks = word_tokenize(clean_text(nostops, removePunct=True))

    # Convert to ids
    ids = tokens_to_ids(toks, vocab)
    
    # Normalize the length
    length = min(maxlen, len(ids))
    ids = normalize_sentence_length(ids, vocab, front_padding=False, length=maxlen, padId=0)
    
    # Return the IDs and length
    return np.asarray(ids), length

def text_to_char_batch(text, vocab, char_vocab, maxlen=20, max_char_len=50, frequency_feature=False):
    '''
    Turn a sentence into word and character tokens based on vocab and lengths

    Args:
        text (String):                  A sentence to convert
        vocab (dict):                   A dictionary mapping tokens to ids
        cahr_vocab (dict):              A dictionary mapping characters to ids
        maxlen (int):                   The token length to normalize the sentence to
        max_char_len (int):             The character length
        frequency_feature (bool):       If true, use character frequencies as opposed to ids
    '''

    # Remove stop words
    nostops = " ".join([t for t in word_tokenize(text) if t not in stop_words])
    # Clean and tokenize
    toks = word_tokenize(clean_text(nostops, removePunct=True))

    # Get character information
    chars_curr = []
    char_lens_curr = []
    if frequency_feature:
        # Get frequency of each character
        counts = collections.Counter(text)
        frequencies = np.zeros(shape=(len(char_vocab),))
        for c in counts:
            if c in char_vocab:
                frequencies[char_vocab[c]] = counts[c]
            else:
                frequencies[char_vocab['<unk>']] = counts[c]
        chars_curr = frequencies
    else:
        # Convert characters to ids
        char_ids = tokens_to_ids([c for c in text], char_vocab)
        char_lens_curr.append(min(max_char_len, len(char_ids)))
        # Normalize lengths
        char_ids = normalize_sentence_length(char_ids, char_vocab, front_padding=False, length=max_char_len, padId=0)
        chars_curr = np.asarray(char_ids)
    
    # Convert tokens to ids and normalize length
    ids = tokens_to_ids(toks, vocab)
    length = min(maxlen, len(ids))
    ids = normalize_sentence_length(ids, vocab, front_padding=False, length=maxlen, padId=0)
    
    return np.asarray(ids), length, chars_curr, char_lens_curr
