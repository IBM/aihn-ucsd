import collections
import logging
import pickle

from nltk.corpus import stopwords

import config

STOPWORDS = set(stopwords.words('english'))

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def process_types(mrsty_file):
    """Reads UMLS semantic types file MRSTY.2019.RRF.
    For details on each column, please check: https://www.ncbi.nlm.nih.gov/books/NBK9685/
    """
    cui_to_entity_types = collections.defaultdict(set)
    with open(mrsty_file) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue

            # Each line is as such:
            # 'C0000005|T116|A1.4.1.2.1.7|Amino Acid, Peptide, or Protein|AT17648347|256|
            # CUI|TUI|STN|STY|ATUI|CVF
            # Unique identifier of concept|Unique identifier of Semantic Type|Semantic Type tree number|Semantic Type. The valid values are defined in the Semantic Network.|Unique identifier for attribute|Content View Flag
            line = line.split("|")

            e_id = line[0]
            e_type = line[3].strip()

            # considering entities with entity types only
            if not e_type:
                continue

            cui_to_entity_types[e_id].add(e_type)

    return cui_to_entity_types


def process_rels(mrrel_file, ro_only=True):
    """Reads UMLS relation triples file MRREL.2019.RRF.

    Use ``ro_only`` to consider relations of "RO" semantic type only.
    50813206 lines in UMLS2019.

    RO = has relationship other than synonymous, narrower, or broader

    For details on each column, please check:
    https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.related_concepts_file_mrrel_rrf/?report=objectonly

    """
    relation_text_to_groups = collections.defaultdict(set)
    with open(mrrel_file) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue

            # Each line is as such: C0012792|A24166664|SCUI|RO|C0026827|A0088733|SCUI|induced_by|R176819430||MED-RT|MED-RT||N|N||
            line = line.split("|")

            # Consider relations of 'RO' type only
            if line[3] != "RO" and ro_only:
                continue

            e1_id = line[0]
            e2_id = line[4]
            rel_text = line[7].strip()

            # considering relations with textual descriptions only
            if not rel_text:
                continue

            relation_text_to_groups[rel_text].add((e1_id, e2_id))
    return relation_text_to_groups


def process_ents(mrconso_file, en_only=True, lower_case=True):
    """Reads UMLS concept names file MRCONSO.2019.RRF.

    Use ``en_only`` to read English concepts only.
    11743183 lines for UMLS2019.

    For details on each column, please check:
    https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/?report=objectonly

    """
    entity_text_to_cuis = collections.defaultdict(set)
    cui_to_entity_texts = collections.defaultdict(set)

    with open(mrconso_file) as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue

            # Each line is as such:  C0000005|ENG|P|L0000005|PF|S0007492|Y|A26634265||M0019694|D012711|MSH|PEP|D012711|(131)I-Macroaggregated Albumin|0|N|256|
            line = line.split("|")

            # Consider en only
            if line[1] != "ENG" and en_only:
                continue

            e_id = line[0]
            e_text = line[-5].strip()

            if not e_text:
                continue

            if lower_case:
                e_text = e_text.lower()

            # Ignore entities with char len = 2
            if len(e_text) <= 2:
                continue
            if e_text.lower() in STOPWORDS:
                continue
            entity_text_to_cuis[e_text].add(e_id)
            cui_to_entity_texts[e_id].add(e_text)
    return entity_text_to_cuis, cui_to_entity_texts


def process_texts_to_type(entity_text_to_cuis, cui_to_entity_types):
    texts_to_type = collections.defaultdict(set)
    for text in entity_text_to_cuis:
        cui = list(entity_text_to_cuis[text])[0]
        texts_to_type[text] = cui_to_entity_types[cui]
    return texts_to_type


def save(fname, data):
    with open(fname, "wb") as wf:
        pickle.dump(data, wf)


def create_UMLSVocab(mrrel_file, mrconso_file, mrsty_file, lower_case=True):
    '''
    Function that converts UMLS data to dictionaries:
    - entity text to CUI
    - CUI to entity text
    - CUI to entity type
    - Relationship text to group (h_cui, t_cui)
    '''
    # Create ent-to-cui and cui-to-ent dictionaries
    logger.info("Reading `{}` for UMLS concepts ...".format(mrconso_file))
    entity_text_to_cuis, cui_to_entity_texts = process_ents(mrconso_file, lower_case=lower_case)
    logger.info("Collected {} unique CUIs and {} unique entities texts.".format(len(cui_to_entity_texts),
                                                                                len(entity_text_to_cuis)))

    # Create cui-to-type dictionary
    logger.info("Reading `{}` for UMLS types ...".format(mrsty_file))
    cui_to_entity_types = process_types(mrsty_file)
    logger.info("Collected {} unique entity types".format(len(cui_to_entity_types)))

    # Create text to type dict
    text_to_type = process_texts_to_type(entity_text_to_cuis, cui_to_entity_types)

    # Create reltxt-to-group dictionary
    logger.info("Reading `{}` for UMLS relations triples ...".format(mrrel_file))
    relation_text_to_groups = process_rels(mrrel_file)

    all_groups = set()
    num_of_triples = 0
    for groups in relation_text_to_groups.values():
        all_groups.update(groups)
        num_of_triples += len(groups)
    num_of_groups = len(all_groups)
    logger.info("Collected {} unique relation texts.".format(len(relation_text_to_groups)))
    logger.info("Collected {} triples with {} unique groups.".format(num_of_triples, num_of_groups))

    # Save each dictionary in its own file
    file_and_data_loc = [
        (config.umls_txt_to_cui, entity_text_to_cuis),
        (config.umls_cui_to_txts, cui_to_entity_texts),
        (config.umls_cui_to_types, cui_to_entity_types),
        (config.umls_text_to_type, text_to_type),
        (config.umls_reltxt_to_groups, relation_text_to_groups)
    ]
    for fname, dictionary in file_and_data_loc:
        save(fname, dictionary)


if __name__ == "__main__":
    create_UMLSVocab(config.mrrel_file, config.mrconso_file, config.mrsty_file)
