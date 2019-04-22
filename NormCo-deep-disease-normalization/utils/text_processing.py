import re
import operator
import string
import numpy as np
import spacy
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))

SENTENCE_SPLIT_REGEX = re.compile(r'\s+')
DOC_SPLIT_REGEX = re.compile(r'[.!?]')
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"


def clean_text(sentence, lower=True, removeSpaces=True, removePunct=False, lowerWithTitles=False):
    """
    Cleans up the input text.

    Args:
        sentence (string):      The input string to clean up
        removePunct (bool):     Whether to remove punctuation

    Returns:
        string: The cleaned string
    """
    if isinstance(sentence, bytes):
        sentence.decode()

    # Remove punctuation
    if removePunct:
        # translator = str.maketrans('', '', string.punctuation)
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        sentence = sentence.translate(translator)

    # Remove multiple spaces
    if removeSpaces:
        sentence = re.sub("\s+", " ", sentence).strip()

    if lower:
        sentence = sentence.lower()
    elif lowerWithTitles and len(sentence) > 1:
        sentence = " ".join([s[0] + s[1:].lower() for s in re.split(SENTENCE_SPLIT_REGEX, sentence)])

    # Strip and return
    return sentence.strip()


def stem_and_lemmatize(tokens, stem=True, lemmatize=True):
    """
    Perform stemming and lemmatization

    Args:
        tokens (list[string]):          List of tokens to stem and lemmatize
        stem (bool):                    If true, perform stemming
        lemmatize (bool):               If True, lemmatize

    Returns:
        The list of tokens stemmed and lemmatized
    
    """
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def word_tokenize(sentence, tokenizer="nltk"):
    """
    Tokenize the input string.

    Args:
        sentence (string):      The input string
        tokenizer (string):     The tokenizer to use. Default is nltk word tokenizer

    Returns:
        List[string]: The tokens from the input string
    """
    if tokenizer in "nltk":
        return nltk_word_tokenize(sentence)
    else:
        return re.split(SENTENCE_SPLIT_REGEX, sentence)


def sent_tokenize(document, tokenizer='nltk'):
    """
    Tokenize the document into a list of sentences.

    Args:
        document (string):      The input string
        tokenizer (string):     The tokenizer to use. Default is nltk sentence tokenizer

    Returns:
        List[string]: The sentences from the input string
    """
    if tokenizer in "nltk":
        return nltk_sent_tokenize(document)
    else:
        return re.split(document, DOC_SPLIT_REGEX)

def conceptTokenize(sentence, tokenizer='nltk'):
    nostops = " ".join([t for t in word_tokenize(sentence) if t not in stop_words])
    if tokenizer in 'nltk':
        return word_tokenize(clean_text(nostops, removePunct=True))
    else:
        return word_tokenize(clean_text(nostops, removePunct=False), tokenizer='spaces')

def getFeatureMap(toks, syns):
    """
    Gets text features for a given piece of text and a list of synonyms

    Args:
        text (List[string]):          A list of tokens
        syns (List[string]):    A list of synonyms
    Returns:
        List[int]: The features present between the given text and synonyms
    """

    word_overlap = 0.0
    acronym = 0.0
    stem_overlap = 0.0
    stemmer = PorterStemmer()
    
    #tokenize the text
    #nostops = " ".join([t for t in word_tokenize(text) if t not in stop_words])
    #toks = word_tokenize(clean_text(nostops, removePunct=True))
    
    for syn_toks in syns:
        acro = ''.join([t[0] for t in syn_toks])

        for t in toks:
            if acro == t:
                acronym = 1.0
                break
                
        for t in toks:
            for s in syn_toks:
                if s == t:
                    word_overlap = 1.0
                elif stemmer.stem(s) == stemmer.stem(t):
                    stem_overlap = 1.0
    return [float(word_overlap), float(stem_overlap), float(acronym)]


def extend_overlapping_spans(spans):
    """
    Method to take a list of spans and extend overlapping spans to the longest span

    Args:
        spans (List[tuple]):    A list of span tuples (start, end)

    Returns:
        List[tuple]: A list of span tuples (start, end) with overlapping spans extended to the longest span
    """
    spans = sorted(spans, key=operator.itemgetter(1))
    i = len(spans) - 1
    while i >= 0:
        start, end = spans[i]
        delete = False
        for j in range(i + 1, len(spans)):
            rstart, rend = spans[j]
            if start < rstart < end:
                spans[j][0] = start
                del spans[i]
                delete = True
                break
            elif start >= rstart:
                del spans[i]
                delete = True
                break
        if ~delete:
            i -= 1
    return spans


def label_tokens_in_spans(text, spans, begin_label="B", in_label='I', out_label="O", label_scheme="BI"):
    """
    Method to label tokens within a span

    Args:
        text (str):          The text to label
        spans (List[tuple]): A list of spans within which to label tokens
        begin_label (str):   The label to use for the first token of a span
        in_label (str):      The label for all tokens inside a span (Ignored if label_scheme="Basic")
        out_label (str):     The label to use for tokens outside of the spans
        label_scheme (str):  The labeling scheme to use options are:
                                - Basic: just use the given label
                                - BI
    Returns:
        List[tuple]: Labeled tokens
    """
    nlp = spacy.blank('en')
    doc = nlp(text)
    labeled = []
    textptr = 0
    for span in spans:
        if span[0] - textptr > 0:
            endI = span[0]
            if text[span[0] - 1] in ' ':
                endI = span[0] - 1
            I = doc.char_span(textptr, endI)
            labeled.extend([(tok.text, out_label) for tok in I])
        # Make sure aligns with word boundaries
        startB = span[0]
        while text[startB] in ' ':
            startB += 1

        # Tokens inside the span
        B = doc.char_span(startB, span[1])
        if label_scheme == 'Basic':
            labeled.extend([(tok.text, begin_label) for tok in B])
        else:
            labeled.append((B[0].text, begin_label))
            labeled.extend([(tok.text, in_label) for tok in B[1:]])
        textptr = span[1]
        while text[textptr] in ' ':
            textptr += 1
    # labeled.extend([(tok, out_label) for tok in word_tokenize(text[textptr:])])
    labeled.extend([(tok.text, out_label) for tok in doc.char_span(textptr, len(text))])

    return labeled


def collect_abstracts(filename):
    with open(filename) as f:
        abstracts = [" ".join(l.split('\t')[1:]).strip() for l in f]
    return abstracts


def load_dict_from_vocab_file(filename):
    """
    Creates a dictionary from a vocabulary file

    Args:
        filename (string): File path to the vocabulary file. Each word should occupy a single line.

    Returns:
        dict{string: int}: The dictionary keyed off of the word with the value being the index
    """
    with open(filename, encoding="utf-8") as f:
        words = [w.strip() for w in f.readlines()]

    return {words[n]: n for n in range(len(words))}


# Code to turn sentences into indices with normalized length
def preprocess_sentence(sentence, vocab, length=10):
    """
    Cleans, tokenizes, and transforms string into an array of indices with normalized length

    Args:
        sentence (string):      The input sentence
        vocab (dict):           The target vocabulary (string, int)
        length (int):           The normalized sentence length or None to maintain original lengths
    
    Returns:
        List[int]: List of word indices
    """
    sentence = clean_text(sentence)
    tokens = word_tokenize(sentence)

    ids = tokens_to_ids(tokens, vocab)
    if length is not None:
        ids = normalize_sentence_length(ids, vocab, length)

    return np.asarray(ids)


def normalize_sentence_length(ids, vocab, front_padding=True, length=10, padId=None):
    """
    Normalize sentences to a given length, filling with a special PAD token

    Args:
        ids (List[int]):        List of token indices
        vocab (dict):           THe target vocabulary (string, int)
        front_padding (bool):   Flag indicating if padding should be applied
                                to the front or back of the string
        length (int):           The target length
        padId (int):            ID to use for padding or None to use vocabulary's pad token
    
    Returns:
        List[int]: List of word indices with normalized length
    """
    if padId is None:
        padId = vocab[PAD_TOKEN]

    if len(ids) > length:
        ids = ids[:length]
    if len(ids) < length:
        if front_padding:
            ids = [padId] * (length - len(ids)) + ids
        else:
            ids = ids + [padId] * (length - len(ids))

    return ids


def tokens_to_ids(tokens, vocab):
    """
    Transform a list of tokens into a list of indices. Out-of-vocab words are mapped
    to a special UNK token

    Args:
        tokens (List[string]):      The list of tokens
        vocab (dict):               The target vocabulary (string, int)

    Returns:
        List[int]: List of word indices in the vocab
    """
    ids = [vocab[t] if t in vocab else vocab[UNK_TOKEN] for t in tokens]

    return ids
