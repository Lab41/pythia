#!/usr/bin/env python

""" Download and parse the Stack Exchange data dump, producing folders of JSON
files for processing by other Pythia algorithms.
"""

import os
from urllib import request, error
import json
from lxml import etree
from bs4 import BeautifulSoup
from src.utils import py7z_extractall
from collections import defaultdict
import argparse
import re

def gen_url(section):
    """URL for a given stackexchange site"""
    urls = []
    urls.append('https://ia800500.us.archive.org/22/items/stackexchange/' + section + '.stackexchange.com.7z')
    urls.append('https://ia800500.us.archive.org/22/items/stackexchange/' + section + '.7z')
    return urls

def get_data(filename):
    """Get URLs of StackExchange sites listed in a file on disk

    Returns:
        list of URLs"""

    #add titles of sections to download
    sections = set()
    with open(filename,'r') as dataFile:
        for line in dataFile: sections.add(line.strip())

    stack_exchange_data = list()
    for section in sections:
        stack_exchange_data.append((section, gen_url(section)))

    return stack_exchange_data

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup(zip_path, dest_path):
    """Create folders, if they don't exist, for
    the zip files from the archive.org release and the processed data.
    """

    #makes folder for zip files
    make_directory(zip_path)

    #makes folder for processed data
    make_directory(dest_path)

def section_setup(section, zip_directory, corpus_directory):
    """Make folders for individual SE site (section)'s unzipped files and
    processed data, and generate expected path to 7z file on disk"""

    #makes folder for unzipped files for a site
    section_directory = os.path.join(zip_directory, section)
    make_directory(section_directory)

    #generate path to release zip file (saved at root zip directory)
    file_name = section + ".7z"
    full_file_name = os.path.join(zip_directory, file_name)

    # Generate folder for processed data
    corpus_section_directory = os.path.join(corpus_directory, section)
    make_directory(corpus_section_directory)

    return full_file_name, section_directory, corpus_section_directory

def load(url, file_name, folder):
    """ Download archive for a StackExchange site and unzip it,
    skipping either or both if the neessary tables are already available """
    # Need special case for Stack Overflow (more than one 7z file)

    if not os.path.isfile(file_name):
        #downloads file from url; two url patterns are attempted
        testfile = request.URLopener()
        try:
            testfile.retrieve(url[0], file_name)
        except error.HTTPError as e:
            try:
                testfile.retrieve(url[1], file_name)
            except:
                print ("Error: URL retrieval of " + url[0] + " and " + url[1] + " failed for reason: " + e.reason)
                quit()

    #un-zips file and puts contents in folder
    a = py7z_extractall.un7zip(file_name)
    if not (os.path.isfile(os.path.join(folder, "PostLinks.xml")) and os.path.isfile(os.path.join(folder, "Posts.xml"))):
        a.extractall(folder)

def get_links(folder):
    """Parse Links table from a SE site data release"""
    tree = etree.parse(folder +"/PostLinks.xml")
    return tree.getroot()

def iter_clusters(links, posts, posthistory):
    related_link = '1'
    duplicate_link = '3'
    closed_duplicates = defaultdict(list)

    for cluster_id, link in enumerate(links):
        dest_id = link.attrib['PostId']
        src_id = link.attrib['RelatedPostId']
        link_type = link.attrib['LinkTypeId']
        if link_type not in (related_link, duplicate_link) or src_id in closed_duplicates.get(dest_id,list()):
            continue
        closed_duplicate = False
        # Determine if dest_id has been closed as a duplicate of src_id. If true, add to closed_duplicates dict
        if link_type == related_link:
            history = posthistory.xpath("//row[@PostId=" + dest_id + " and @PostHistoryTypeId=10" \
                                        + " and (@Comment=1 or @Comment=101) " + "]")
            if len(history)>0:
                for entry in history:
                    start = entry.attrib['Text'].find("OriginalQuestionIds") + 22
                    end = entry.attrib['Text'].find("]",start)
                    question_ids = entry.attrib['Text'][start:end].split(',')
                    if src_id in question_ids:
                        closed_duplicates[dest_id].append(src_id)
                        closed_duplicate = True

        src_text = extract_post_text(src_id, posts)
        dest_text = extract_post_text(dest_id, posts)
        if src_text is None or dest_text is None:
            continue

        src_doc = { 'post_id': src_id,
            'order' : 0,
            'body_text' : src_text,
            'novelty' : True,
            'cluster_id' : cluster_id
        }
        dest_doc = { 'post_id': dest_id,
            'order' : 1,
            'body_text' : dest_text,
            'novelty' : True if (link_type == related_link and not closed_duplicate) else False,
            'cluster_id' : cluster_id
        }
        yield src_doc, dest_doc

def gen_clusters(links, posts):
    """
    Given links, return a data structure representing ordered lists of documents
    with associated metadata and novelty markings.

    Only outputs lists of two directly linked documents due to lingering
    questions about how to interpret indirect links.

    Args:
        links (list): as from get_links

    Returns:
        list of list of dict -- the clusters, each one a list of dicts
        representing the document objects
    """

    clusters = list(iter_clusters(links, posts))
    return clusters

def get_post_history(folder):
    tree = etree.parse(folder +"/PostHistory.xml")
    return tree.getroot()

def get_posts(folder):
    tree = etree.parse(folder +"/Posts.xml")
    return tree.getroot()

def clean_up(raw_text, isbody):
    # Remove duplicate warning text
    if isbody and raw_text.find('Possible Duplicate'):
        raw_text = re.sub(r"<\/?blockquote>.*<\/?blockquote>(.*)", r"\1", raw_text, flags=re.MULTILINE | re.DOTALL)
    return BeautifulSoup(raw_text, "lxml").get_text()

def extract_post_text(id, posts):
    """
        Get post by ID from XML etree object

        Args:
            id (int): ID of post to retrieve
            posts (lxml.etree._Element): root of parsed posts table
    """
    try:
        post = posts.find("./*[@Id='{id}']".format(id=id))
        return clean_up(post.attrib['Title'],False) + ' ' + clean_up(post.attrib['Body'],True)
    except AttributeError:
        return None
    except KeyError:
        return None

def write_json_files(clusters, corpus_directory, filename_prefix=''):
    for cluster in clusters:
        cluster_id = cluster[0]['cluster_id']
        cluster_filename = "{}{:05d}.json".format(filename_prefix, cluster_id)
        cluster_path = os.path.join(corpus_directory, cluster_filename)
        with open(cluster_path, "w") as cluster_out:
            for doc in cluster:
                print(json.dumps(doc), file=cluster_out)

def main(args):
    """Go through list of SE sites, create directories to store downloaded data
    and parsed clusters. Process SE data releases into Pythia-specific format.

    Arguments:
        args (namespace): from parser, below
    """

    #gets urls based on sections and creates basic directories
    stack_exchange_data = get_data(args.filename)
    zip_directory, corpus_directory = args.zip_path, args.dest_path
    setup(zip_directory, corpus_directory)

    for (section, url) in stack_exchange_data:
        #creates directories for the current SE site
        zip_file_path, unzipped_folder, corpus_section_directory = section_setup(
            section, zip_directory, corpus_directory)

        done_signal_path = os.path.join(corpus_section_directory, ".done")
        if os.path.isfile(done_signal_path):
            continue

        print("Starting " + section)

        #downloads and unzips data release for a site
        load(url, zip_file_path, unzipped_folder)

        #gets the links data from the links table for the site
        links = get_links(unzipped_folder)

        #gets post data from the posts table
        posts = get_posts(unzipped_folder)

        #gets post history
        posthistory = get_post_history(unzipped_folder)

        #creates the clusters of related and duplicate posts for a site,
        #based on links data
        # clusters, related, duplicates, unique_posts = gen_clusters(links)
        clusters = iter_clusters(links, posts, posthistory)

        #writes cluster information to json files
        write_json_files(clusters, corpus_section_directory)
        
        # put completion marker in folder so we can skip it next time
        with open(done_signal_path, "w") as f:
            print("", file=f)

        print("Completed " + section)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Parse Stack Exchange user " \
        "posts into JSON files for use in Pythia project")
    parser.add_argument("filename", help="file containing list of Stack " \
        "Exchange sites (ex: astronomy) to download/parse")
    parser.add_argument("--zip-path", help="path to folder where zip files are " \
        "located; files will be downloaded here if they don't exist",
        default="stack_exchange_data/zip_files")
    parser.add_argument("--dest-path", help="path to folder where processed data will " \
        "be stored.", default="stack_exchange_data/corpus")
    args = parser.parse_args()
    main(args)
    #parser.exit(status=0, message=None)
