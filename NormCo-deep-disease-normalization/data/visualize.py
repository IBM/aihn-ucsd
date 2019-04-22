import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

def getMentions(file, pmid_list=None):
    mentions = []
    with open(file) as f:
        curr_mentions = []
        for l in f:
            if l in '\n':
                if pmid_list is not None and len(curr_mentions) > 0:
                    pmid = curr_mentions[0].split('|')[0]
                    if pmid in pmid_list:
                        mentions.append(curr_mentions)
                elif len(curr_mentions) > 0:       
                    mentions.append(curr_mentions)
                curr_mentions = []
            else:
                curr_mentions.append(l)
        if len(curr_mentions) > 1:
            if pmid_list is not None:
                pmid = curr_mentions[0].split('|')[0]
                if pmid in pmid_list:
                    mentions.append(curr_mentions)
            else:       
                mentions.append(curr_mentions)
    return mentions

def getAbstractTextFromMentions(mentions):
    abstracts = {}
    for m in mentions:
        pmid = m[0].split('|')[0]
        abstracts[pmid] = m[0].split('|')[2] + m[1].split('|')[2]
    return abstracts

def groupErrorsByAbstract(errors):
    error_dict = defaultdict(list)
    for e in errors.values:
        error_dict[e[0]].append(e)
    return error_dict


if __name__ == "__main__":
    ################################################
    # Parse command line arguments
    ################################################
    parser = argparse.ArgumentParser(description="Train the embedding model to embed synonyms close to each other")
    parser.add_argument('--abstracts_file', type=str, help='The location of the abstracts', required=True)
    parser.add_argument('--pmids_file', type=str, help='The location of the pmids of the abstracts to use', default=None)
    parser.add_argument('--hierarchy_file', type=str, help='Location of the ontology hierarchy', required=True)
    parser.add_argument('--error_file', type=str, help='Location of output error analysis file', required=True)
    parser.add_argument('--disease_dict', type=str, help='The location of the disease dictionary', default=None)
    parser.add_argument('--output_file', type=str, help='Location of output html file', required=True)
    
    args = parser.parse_args()

    disease_dict = pd.read_csv(args.disease_dict, sep='\t', header=None, comment="#").fillna('')
    errors = pd.read_csv(args.error_file, sep='\t').fillna('')
    aggregated_errors = errors.groupby(['pmid', 'span_start', 'span_end', 'actual', 'actual_preferred', 'prediction', 'prediction_preferred'], as_index=False).agg('|'.join)

    pmid_list = None
    if args.pmids_file is not None:
        with open(pmids_file) as f:
            pmid_list = set([l.strip() for l in f])
    mentions = getMentions(args.abstracts_file, pmid_list)
    abstracts = getAbstractTextFromMentions(mentions)
    errors_by_abstract = groupErrorsByAbstract(aggregated_errors)

    tree_map = defaultdict(dict)
    with open(args.hierarchy_file) as f:
        for l in f:
            fields = l[:-1].split('\t')
            tree_map[fields[0]][fields[1]] = int(fields[2])

    count = 0
    html_output = '<html><body>'
    for ab in errors_by_abstract:
        abstract = abstracts[str(ab)]
        start = 0
        last_start = -1
        markdown = '<strong>' + str(ab) + '</strong> '
        for m in errors_by_abstract[ab]:
            if last_start == m[1]:
                continue
            markdown += abstract[start:m[1]]
            color = 'green'
            fam_text = ''
            if m[3] != m[5]:
                color = 'red'
                fam_text = ', Not in family'
                if m[5] in tree_map[m[3]]:
                    fam_text = ', TOO SPECIFIC ' if tree_map[m[3]][m[5]] > 0 else ', TOO GENERAL '
                    fam_text += "(" + str(tree_map[m[3]][m[5]]) + ")"
            markdown += '<strong><abbr title="Guessed: ' + m[5] + ' (' + m[6] + '), Correct: ' + m[3] + ' (' + m[4] + ')' + fam_text + '"><span style="color:' + color + '">'
            markdown += abstract[m[1]:m[2]]
            markdown += "</span></abbr></strong>"
            start = m[2]
            last_start = m[1]
            count += 1
        markdown += abstract[start:]
        html_output += '<p>' + markdown + '</p><br/>'
    html_output += '</body></html>'

    with open(args.output_file, 'w') as f:
        f.write(html_output)
