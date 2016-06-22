#!/usr/bin/env python 

import sys
import os
import py7zlib

class un7zip(object):
    @classmethod
    def is_7zfile(cls, filepath):
        '''
        Class method: determine if file path points to a valid 7z archive.
        '''
        is7z = False
        fp = None
        try:
            print("opening "+filepath)
            fp = open(filepath, 'rb')
            archive = py7zlib.Archive7z(fp)
            n = len(archive.getnames())
            is7z = True
            print("7zip file")
        finally:
            if fp:
                fp.close()
        return is7z
    def __init__(self, filepath):
        fp = open(filepath, 'rb')
        self.archive = py7zlib.Archive7z(fp)
    def extractall(self, path):
        for name in self.archive.getnames():
            outfilename = os.path.join(path, name)
            outdir = os.path.dirname(outfilename)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outfile = open(outfilename, 'wb')
            outfile.write(self.archive.getmember(name).read())
            outfile.close()

if __name__=="__main__":
    a = un7zip(sys.argv[1])
    a.extractall(".")
