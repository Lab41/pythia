import os
from urllib import request
import subprocess
import json
from lxml import etree
from bs4 import BeautifulSoup
from src.utils import py7z_extractall

def gen_url(section):
    return 'https://ia800500.us.archive.org/22/items/stackexchange/' + section + '.stackexchange.com.7z'

def get_data():

    #add titles of sections to download
    sections = {'3dprinting', 'academia', 'android', 'anime', 'apple', 'arabic', 'arduino', 'sports'}

    stack_exchange_data = list()
    for section in sections:
        stack_exchange_data.append((section, gen_url(section)))

    return stack_exchange_data

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup():

    #makes folder for all stack exchange data
    directory = 'stack_exchange_data'
    make_directory(directory)

    #makes folder for zip files
    zip_directory = os.path.join(directory, 'zip_files')
    make_directory(zip_directory)

    #makes folder for corpus
    corpus_directory = os.path.join(directory, 'corpus')
    make_directory(corpus_directory)

    return directory, zip_directory, corpus_directory

def section_setup(section, directory, zip_directory, corpus_directory):

    #makes folder for section
    section_directory = os.path.join(directory, section + "_files")
    make_directory(section_directory)

    #info for section files
    file_name = section + "_stackexchange.7z"
    full_file_name = os.path.join(zip_directory, file_name)

    corpus_section_directory = os.path.join(corpus_directory, section)
    make_directory(corpus_section_directory)

    return full_file_name, section_directory, corpus_section_directory

def load(url, file_name, folder):

    #downloads file from url
    testfile = request.URLopener()
    testfile.retrieve(url, file_name)

    #un-zips file and puts contents in folder
    a = py7z_extractall.un7zip(file_name)
    a.extractall(folder)

def get_links(folder):
    tree = etree.parse(folder +"/PostLinks.xml")
    return tree.getroot()

def gen_clusters(links):
    unused_id = 1

    related_link = '1'
    duplicate_link = '3'

    clusters = dict()
    related = dict()
    duplicates = dict()
    unique_posts = set()

    for l in links:
        post_id = l.attrib['PostId']
        related_id = l.attrib['RelatedPostId']
        new_ids = {post_id, related_id}
        unique_posts = unique_posts.union(new_ids)
        post_cluster_id = None
        related_cluster_id = None
        for c in clusters:
            ids = clusters[c]
            if post_id in ids:
                post_cluster_id = c
            elif related_id in ids:
                related_cluster_id = c
        if not (post_cluster_id or related_cluster_id):
            cluster_id = unused_id
            clusters[cluster_id] = set()
            duplicates[cluster_id] = set()
            related[cluster_id] = set()
            unused_id+=1
        elif not related_cluster_id:
            cluster_id = post_cluster_id
        elif not post_cluster_id:
            cluster_id = related_cluster_id
        else: #both ids appeared in clusters
            post_cluster = clusters[post_cluster_id]
            related_cluster = clusters[related_cluster_id]
            clusters[post_cluster_id] = post_cluster.union(related_cluster)
            del clusters[related_cluster_id]
            cluster_id = post_cluster_id
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
                corpus[id] = clean_up(p.attrib['Title']) #+ ' ' + clean_up(p.attrib['Body'])
            except:
                pass
    return corpus

def write_json_files(clusters, related, duplicates, corpus, corpus_directory):
    next_cluster_id = 0
    for cluster_id in clusters:
        time_stamp = 0
        file_empty = True
        file_name = str(next_cluster_id) + '.json'
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

def main(argv):

    #gets urls based on sections and creates basic directories
    stack_exchange_data = get_data()
    directory, zip_directory, corpus_directory = setup()

    for (section, url) in stack_exchange_data:

        #creates directorys for section
        file_name, folder_name, corpus_section_directory = section_setup(section, directory, zip_directory, corpus_directory)

        #downloads and unzips section file
        load(url, file_name, folder_name)

        #gets the links from the links file
        links = get_links(folder_name)

        #creates the clusters
        clusters, related, duplicates, unique_posts = gen_clusters(links)

        #gets the posts from the posts file
        posts = get_posts(folder_name)

        #creates the corpus with the body text for each id in the clusters
        corpus = gen_corpus(posts, unique_posts)

        #writes the information to json files
        write_json_files(clusters, related, duplicates, corpus, corpus_section_directory)

if __name__ == '__main__':
    main(None)
