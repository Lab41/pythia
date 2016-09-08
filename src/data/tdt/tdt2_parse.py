#!/usr/bin/env python
"""
tdt_parse.py

Parse TDT2 formatted topics and docs from a given base directory and generate Pythia JSON formatted files.
"""

import os
import json
from bs4 import BeautifulSoup as bsoup
import argparse
import logging
from datetime import datetime,timedelta
logger = logging.getLogger("tdt_parse")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)

DEFAULT_DOCTYPE="NEWS"
# TDT SGML tags for doc entries
DOC_KEY = "DOC"
DOCTYPE_KEY = "DOCTYPE"
DOCNO_KEY = "DOCNO"
TEXT_KEY = "TEXT"
ENG_ENCODING = 'UTF-8'
MAN_ENCODING = 'GB18030'
ENGLISH_SRCS = ('NYT_NYT', 'APW_ENG', 'CNN_HDL', 'ABC_WNT', 'PRI_TWD', 'VOA_ENG')
MANDARIN_SRCS = ('XIN_MAN', 'ZBN_MAN', 'VOA_MAN')

# Pythia JSON 
CLUSTER_ID = 'cluster_id'
POST_ID = 'post_id'
ORDER = 'order'
BODY_TEXT = 'body_text'
NOVELTY = 'novelty'
# document type to be converted and saved as Pythia XML
# TODO: allow for multiple DOCTYPE values?
DEFAULT_SRC_DIR = './tdt2_e_v4_0/'
DEFAULT_DEST_DIR = './pythia_tdt/'
REL_TOPIC_DIR = 'doc/topics/'
TDT_DOC_DIR = 'tdt2_em/tkn_sgm/'
TDT_DOC_EXT = '.tkn_sgm'
TOPIC_REL_FILE = 'tdt2_topic_rel.complete_annot.v3.3'

def parse_tdt_by_topic(src_dir, doc_type, limit = 0, lang = None):
    """
    Iterate over the complete list topics from the given file and parse as a dictionary of 
    Pythia clusters, mapped to an array of relevant docs.
    """
    logger.info('parse_tdt_topics(%s, %s)', src_dir, doc_type)
    topic_file = os.path.join(src_dir, REL_TOPIC_DIR, TOPIC_REL_FILE)
    clusters = dict()
    count = 0
    with open(topic_file) as fin:
        for line in fin:
            count+=1
            if (limit > 0) and (count >= limit): 
                logger.info('Limit of %s documents reached.', limit)
                break
            ontopic = bsoup(line, 'lxml').ontopic
            logger.debug(ontopic)
            if ontopic is not None:
                tdt_level = ontopic['level']
                # Not considering stories with only BRIEF topic references
                if 'BRIEF' == tdt_level:
                    continue
                post_id = ontopic['docno']
                tdt_topicid = ontopic['topicid']
                tdt_fileid = ontopic['fileid']
                doc_date = tdt_fileid.split('_')[0]
                doc_src = "_".join(tdt_fileid.split('_')[-2:])
                # If a language was specified, limit to sources in the given language
                if lang is not None:
                    if 'ENG' == lang and (doc_src not in ENGLISH_SRCS):
                        logger.debug("Skipping non-English source document.")
                        continue
                    if 'MAN' == lang and (doc_src not in MANDARIN_SRCS):
                        logger.debug("Skipping non-Mandarin source document.")
                        continue
                cluster_id = '{topic}_{date}'.format(topic=tdt_topicid,date=doc_date)
                cluster = clusters.get(cluster_id, dict())
                post = cluster.get(post_id, dict({'post_id':post_id}))
                post['tdt_level'] = tdt_level
                post['novelty'] = False
                if len(cluster) == 0:
                    post['novelty'] = True
#                if 'BRIEF' == tdt_level: 
#                    post['novelty'] = False
                # FIXME - determine a realistic way to assign novelty
#                post['novelty'] = 'TBD'
                post['cluster_id'] = cluster_id
                post['tdt_topicid'] = tdt_topicid
                # TODO: look at alternatives for settign order, e.g. timestamp
                post['order'] = len(cluster)
                post['tdt_fileid'] = tdt_fileid
                post['tdt_src'] = doc_src
                # TODO - get text from source file and add as 'body_text'
                post['body_text'] = extract_doc_text(src_dir, tdt_fileid, post_id)
                cluster[post_id] = post
                clusters[cluster_id] = cluster
#                 logger.debug("Doc:")
#                 logger.debug(cluster[post_id])
    return clusters

def extract_doc_text(src_dir, file_id, doc_id):
    """
    Extracts and returns the TEXT portion for the given doc_id, using the given file_id.
    """
    logger.debug("extract_doc_text(%s, %s) called...", file_id, doc_id)
    doc_text = ""
    doc_src = "_".join(file_id.split('_')[-2:])
    doc_encoding = ENG_ENCODING
    if doc_src in MANDARIN_SRCS:
        doc_encoding = MAN_ENCODING
    # TODO: Make paths OS agnostic
    fname = os.path.join(src_dir,TDT_DOC_DIR,'{dfile}{ext}'.format(dfile=file_id,ext=TDT_DOC_EXT))
    logger.debug("TDT doc file: %s", fname)
    with open(fname, encoding=doc_encoding) as fin:
        file_data = fin.read()
        docs = bsoup(file_data, 'lxml', from_encoding=doc_encoding)
        docs = docs.find_all('doc')
        for doc in docs:
            if doc.docno.get_text(strip=True) == doc_id:
                logger.debug("Extracting text...")
                doc_text = doc.find('text').string
    return doc_text


def make_directory(directory):
    """
    Make a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_all(clusters, dest_dir):
    logger.info("writing clusters as JSON to %s.", dest_dir)
    for cluster_id, cluster in clusters.items():
        logger.debug(cluster_id)
        dest_file = '{dir}{fname}.json'.format(dir=dest_dir,fname=cluster_id)
        # Only write out clusters that contain more than one document
        if len(cluster.values()) > 1:
            write_json(dest_file, cluster)

def write_json(dest_file, docs):
    """
    Write the JSON data out to a file.
    """
    with open(dest_file, 'w') as fout:
        for doc in docs.values():
            json.dump(doc, fout)
            fout.write('\n')
    return

""" Defines a novelty label for the parsed clusters. For a given cluster, finds a new document on the same topic
that occurred some number of days later. Marks the new document as novel and merges it with original cluster.

Args:
 clusters (dict): - TDT news stories grouped by topic and date
 mindays (int): - Minimum number of days between cluster and new document
 maxdays (int): Maximum number of days between cluster and new document
 lang (str): Filter for document language (English or Mandarin)
Output:
 newclusters(dict) - TDT news stories grouped by topic and date with one new document from later date
"""
def define_novelty(clusters, mindays, maxdays, lang):
    newclusters = dict()

    for cluster in clusters:

        # Only keep original clusters with more than one story
        post_id = clusters[cluster]
        post_id_length = len(post_id.values())
        if len(post_id.values()) == 1:
            continue

        # Determine cluster topic and date. Cluster is formatted as topicid_YYYYMMDD
        topic, firstdatestring = cluster.split("_")
        firstdate = datetime.strptime(firstdatestring,'%Y%m%d')

        # Iterate over date range supplied at command line
        for numdays in range(mindays, maxdays+1):

            # Determine future date and look for matching file in clusters
            seconddate = firstdate + timedelta(days=numdays)
            seconddatestring = str(seconddate.strftime('%Y%m%d'))
            try:
                futurecluster = clusters[topic+"_"+ seconddatestring]
            except KeyError:
                continue

            # Make a new cluster from original cluster elements and new future story marked as novel
            suffix = 1
            for entry in futurecluster:

                # Set new cluster label as topicid_YYYYMMDD_YYYYMMDD_lang_iteration
                newclusterlabel = cluster + "_" + seconddatestring + "_" + lang + "_" + str(suffix).zfill(4)
                newclusters[newclusterlabel] = dict()

                # Add existing cluster documents and update cluster label
                for item in post_id:
                    newclusters[newclusterlabel][item] = dict(post_id[item])
                    newclusters[newclusterlabel][item]['cluster_id'] = newclusterlabel

                # Add future document
                newclusters[newclusterlabel][entry] = dict(futurecluster[entry])
                # Set new document to end of new cluster order
                newclusters[newclusterlabel][entry]['order'] = post_id_length
                # Set new document novelty to True
                newclusters[newclusterlabel][entry]['novelty'] = True
                # Update cluster label for new document
                newclusters[newclusterlabel][entry]['cluster_id'] = newclusterlabel

                suffix +=1

    return newclusters

def main(args):
    """
    Parse through the given 
    """
    doc_type = DEFAULT_DOCTYPE
    src_dir = DEFAULT_SRC_DIR
    dest_dir = DEFAULT_DEST_DIR
    limit = 0
    lang = None
    if args.sourcedir:
        logger.debug("Setting src_dir to %s.", args.sourcedir)
        src_dir = args.sourcedir
        if not src_dir.endswith('/'):
            src_dir = "".join([src_dir, '/'])
    if args.destdir:
        logger.debug("Setting dest_dir to %s.", args.destdir)
        dest_dir = args.destdir
        if not dest_dir.endswith('/'):
            dest_dir = "".join([dest_dir, '/'])
    # verify dest directory exists
    make_directory(dest_dir)
    if args.doctype:
        logger.debug("Setting doctype to %s.", args.doctype)
        doc_type = args.doctype
    if args.limit:
        logger.debug("Setting limit to %s.", args.limit)
        limit = args.limit
    if args.lang:
        logger.debug("Setting lang to %s.", args.lang)
        lang = args.lang
    logger.debug("tdt2_parse translating TDT2 data from %s to JSON formatted Pythia data in %s.", src_dir, dest_dir)
    clusters = parse_tdt_by_topic(src_dir, doc_type, limit, lang)
    newclusters = define_novelty(clusters, args.mindays, args.maxdays, args.lang)
    write_all(newclusters, dest_dir)
    return


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description = "Parse TDT data into JSON files for use in Pythia project")
    parser.add_argument("--sourcedir", help="Base TDT2 directory containing the extracted directories containing the formatted files to be parsed. Default is ./tdt2_e_v4_0/")
    parser.add_argument("--destdir", help="Directory to store Pythia JSON formatted files. Default is ./pythia_tdt/")
    parser.add_argument("--doctype", help="Only convert documents of the given DOCTYPE. Currently only supports parsing NEWS, which is the default value")
    parser.add_argument("--limit", type=int, help="Limit the number of files that are processed.")
    parser.add_argument('--lang', default='ENG', help="Limit parsed documents to language, either ENG for english or MAN for Mandarin Chinese (default is 'ENG').")
    parser.add_argument('--mindays', type=int, default=7, help="Minimum number of days of separation between a topic cluster and a new document on the same topic (default is 5). This is used for novelty labeling.")
    parser.add_argument('--maxdays', type=int, default=10, help="Maximum number of days of separation between a topic cluster and a new document on the same topic (default is 10). This is used for novelty labeling.")
    args = parser.parse_args()
    if args.lang not in {'ENG','MAN'}:
        logger.error("Language parameter (--lang) must be 'ENG' (English) or 'MAN' (Mandarin)")
        quit()
    main(args)
    parser.exit(status=0, message=None)
