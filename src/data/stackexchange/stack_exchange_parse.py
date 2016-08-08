#!/usr/bin/env python

""" Download and parse the Stack Exchange data dump, producing folders of JSON
files for processing by other Pythia algorithms.
"""

import os
from urllib import request, error
import subprocess
import json
from lxml import etree
from bs4 import BeautifulSoup
from src.utils import py7z_extractall
from collections import namedtuple
from shutil import copy
import argparse

def gen_url(section):
    """URL for a given stackexchange site"""
    return 'https://ia800500.us.archive.org/22/items/stackexchange/' + section + '.stackexchange.com.7z'

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
    # Need special case for Stack Overflow (more than one 7z file)

    if not os.path.isfile(file_name):
        #downloads file from url
        testfile = request.URLopener()
        try:
            testfile.retrieve(url, file_name)
        except error.HTTPError as e:
            print ("Error: URL retrieval of " + url + " failed for reason: " + e.reason)
            quit()

    #un-zips file and puts contents in folder
    a = py7z_extractall.un7zip(file_name)
    a.extractall(folder)

def get_links(folder):
    """Parse Links table from a SE site data release"""
    tree = etree.parse(folder +"/PostLinks.xml")
    return tree.getroot()

def gen_clusters(links):
    """Find clusters of associated posts using data about duplication and
    relatedness from the links tables of the SE data dump.

    Returns: tuple--
        clusters (dict): mapping of cluster id to set of unique documents in cluster
        related (dict): mapping of cluster id to set of unique documents in cluster
            linked by "related" links
        duplicates (dict): mapping of cluster id to set of unique documents in
            cluster linked by "duplicate" links
        unique_posts (set): set of all unique post ids encountered
    """
    unused_id = 1

    # Code numbers for links in SE Links table
    related_link = '1'
    duplicate_link = '3'

    clusters = dict()
    related = dict()
    duplicates = dict()
    unique_posts = set()

    for l in links:
        # src and destination for a given link
        post_id = l.attrib['PostId']
        related_id = l.attrib['RelatedPostId']
        new_ids = {post_id, related_id}
        # maintain list of post ids encountered
        unique_posts = unique_posts.union(new_ids)
        post_cluster_id = None
        related_cluster_id = None
        # add related posts to existing clusters or create new ones
        for c in clusters:
            ids = clusters[c]
            # does src or dest occur in this cluster? if so, add here
            if post_id in ids:
                post_cluster_id = c
            elif related_id in ids:
                related_cluster_id = c
        # create new cluster and augment index of next cluster, if neither
        # src nor dest was a match
        if not (post_cluster_id or related_cluster_id):
            cluster_id = unused_id
            clusters[cluster_id] = set()
            duplicates[cluster_id] = set()
            related[cluster_id] = set()
            unused_id+=1
        # use cluster assigned to src, if dest not a match
        elif not related_cluster_id:
            cluster_id = post_cluster_id
        # use cluster assigned to dest, if src not a match
        elif not post_cluster_id:
            cluster_id = related_cluster_id
        # or merge clusters, if both src and dest matched clusters
        # if post_cluster_id and related_cluster_id are
        # already the same, merge is harmless
        else:
            post_cluster = clusters[post_cluster_id]
            related_cluster = clusters[related_cluster_id]
            clusters[post_cluster_id] = post_cluster.union(related_cluster)
            del clusters[related_cluster_id]
            cluster_id = post_cluster_id

        # Having found the appropriate cluster id, make sure src and dest
        # are added to it, and data on whether they are linked by relatedness
        # or duplication
        clusters[cluster_id] = clusters[cluster_id].union(new_ids)
        if l.attrib['LinkTypeId'] == related_link:
            related[cluster_id] = related[cluster_id].union(new_ids)
        else: # l.attrib['LinkTypeId'] == duplicate:
            if not l.attrib['LinkTypeId'] == duplicate_link:
                print(l.attrib['LinkTypeId'])
            duplicates[cluster_id] = clusters[cluster_id].union(new_ids)
    return clusters, related, duplicates, unique_posts

def get_posts(folder):
    tree = etree.parse(folder +"/Posts.xml")
    return tree.getroot()

def clean_up(raw_text):
    return BeautifulSoup(raw_text, "lxml").get_text()

def gen_corpus(posts, unique_posts):
    corpus = dict()

    for p in posts:
        id = p.attrib['Id']
        if id in unique_posts:
            try:
                corpus[id] = clean_up(p.attrib['Title']) + ' ' + clean_up(p.attrib['Body'])
            except:
                pass
    return corpus

def write_json_files(clusters, related, duplicates, corpus, corpus_directory):
    next_cluster_id = 0
    for cluster_id in clusters:
        time_stamp = 0
        file_empty = True
        file_name = '{:05d}.json'.format(next_cluster_id)
        full_file_name = os.path.join(corpus_directory, file_name)
        with open(full_file_name, 'w') as outfile:
            if cluster_id in duplicates:
                novel = True
                for duplicate in duplicates[cluster_id]:
                    if duplicate in corpus:
                        d = dict()
                        d['cluster_id'] = next_cluster_id
                        d['post_id'] = duplicate
                        d['order'] = time_stamp
                        d['body_text'] = corpus[duplicate]
                        d['novelty'] = novel
                        json.dump(d, outfile)
                        outfile.write('\n')
                        novel = False
                        time_stamp+=1
                        file_empty = False
            for related_post in related[cluster_id]:
                if not related_post in duplicates:
                    if related_post in corpus:
                        r = dict()
                        r['cluster_id'] = next_cluster_id
                        r['post_id'] = related_post
                        r['order'] = time_stamp
                        r['body_text'] = corpus[related_post]
                        r['novelty'] = True
                        json.dump(r, outfile)
                        outfile.write('\n')
                        time_stamp+=1
                        file_empty = False
        if not file_empty:
            next_cluster_id+=1

def filter_json_files(filtered_corpus_directory, corpus_directory, minpost, maxpost):

    print("Filtering JSON files")
    make_directory(filtered_corpus_directory)

    filestokeep = list()

    # Iterate over topic folders in corpus
    for foldername in os.listdir(corpus_directory):

        fullfoldername = os.path.join(corpus_directory,foldername)

        if os.path.isdir(fullfoldername) == True:

            jsonstats = []

            # Iterate over clusters in this topic
            for file_name in os.listdir(fullfoldername):
                if file_name.endswith(".json"):
                    full_file_name = os.path.join(fullfoldername, file_name)
                    entries = 0
                    with open(full_file_name,'r') as dataFile:
                        for line in dataFile: entries += 1
                    if entries >= minpost and entries <= maxpost: filestokeep.append((full_file_name, foldername))

    # Copy cluster files that meet min and max post requirements
    for entry in filestokeep:
       copylocation = os.path.join(filtered_corpus_directory, entry[1])
       make_directory(copylocation)
       copy(entry[0], copylocation)

    print("Filtered corpus copied to: ", filtered_corpus_directory)

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

    if args.skipparse == False:
        for (section, url) in stack_exchange_data:
            print("Starting " + section)

            #creates directories for the current SE site
            zip_file_path, unzipped_folder, corpus_section_directory = section_setup(
                section, zip_directory, corpus_directory)

            #downloads and unzips data release for a site
            load(url, zip_file_path, unzipped_folder)

            #gets the links data from the links table for the site
            links = get_links(unzipped_folder)

            #creates the clusters of related and duplicate posts for a site,
            #based on links data
            clusters, related, duplicates, unique_posts = gen_clusters(links)

            #gets post data from the posts table
            posts = get_posts(unzipped_folder)

            #extract post title and body text for each document in the site
            corpus = gen_corpus(posts, unique_posts)

            #writes cluster information to json files
            write_json_files(clusters, related, duplicates, corpus, corpus_section_directory)

            print("Completed " + section)

    if args.filter or args.skipparse:
        filter_json_files(os.path.normpath(corpus_directory) + "_filtered",
            corpus_directory, int(args.minpost), int(args.maxpost))


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
    parser.add_argument("--filter", help="flag to filter JSON files after " \
        "downloading/parsing Stack Exchange data, based on minpost/maxpost arguments",
        action="store_true")
    parser.add_argument("--minpost", default=3, help="when filtering, set " \
        "minimum allowable posts in a single JSON file (default is 3)")
    parser.add_argument("--maxpost", default=10, help="when filtering, set " \
        "maximum allowable posts in a single JSON file (default is 10)")
    parser.add_argument("--skipparse", help="flag to bypass downloading/parsing " \
        "JSON files and proceed directly to JSON file filtering; " \
        "can be used if corpus was previously downloaded/parsed", action="store_true")

    args = parser.parse_args()
    main(args)
    #parser.exit(status=0, message=None)
