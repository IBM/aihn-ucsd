import argparse
import hashlib
import logging
import time

import spacy

import config

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MEDLINESpacySents:

    def __init__(self, medline_abstracts, output_fname):
        self.medline_abstracts = medline_abstracts
        self.output_fname = output_fname
        self.sent_tok = spacy.load("en_core_sci_lg")
        logger.info("Using spacy.")

    def extract_sentences(self):
        n, d = 0, 0
        logger.info("Extracting sentences from `{}` ...".format(self.medline_abstracts))
        hash_set = set()
        with open(self.medline_abstracts, encoding="utf-8", errors="ignore") as rf, open(self.output_fname, "w") as wf:
            for idx, abstract in enumerate(rf):
                d += 1
                abstract = abstract.strip()
                if not abstract:
                    continue
                # Strip starting b' or b" and ending ' or "
                if (abstract[:2] == "b'" and abstract[-1] == "'") or (abstract[:2] == 'b"' and abstract[-1] == '"'):
                    abstract = abstract[2:-1]
                for sent in self.sent_tok(abstract).sents:
                    sent = sent.text
                    shash = hashlib.sha256(sent.encode("utf-8")).hexdigest()
                    if shash not in hash_set:
                        hash_set.add(shash)
                        wf.write(sent + "\n")


if __name__ == "__main__":
    infile = config.medline_file
    outfile = config.medline_spacy_sents
    print("Infile {}, Outfile {}".format(infile, outfile))
    ms = MEDLINESpacySents(infile, outfile)
    t = time.time()
    ms.extract_sentences()
    t = (time.time() - t) // 60
    logger.info("Took {} mins!".format(t))
