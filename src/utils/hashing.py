
import hashlib, os
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def dir_hash(directory, verbose=0):
    """# http://akiscode.com/articles/sha-1directoryhash.shtml
    # Copyright (c) 2009 Stephen Akiki
    # MIT License (Means you can do whatever you want with this)
    #  See http://www.opensource.org/licenses/mit-license.php
    # Error Codes:
    #   -1 -> Directory does not exist
    #   -2 -> General error (see stack traceback)"""
    SHAhash = hashlib.sha1()
    if not os.path.exists (directory):
        return -1
    
    try:
        for root, dirs, files in os.walk(directory):
            for names in files:
                if verbose == 1:
                    logger.info('Hashing', names)
                filepath = os.path.join(root,names)
                try:
                     f1 = open(filepath, 'rb')
                except:
                     # You can't open the file for some reason
                     f1.close()
                     continue

                while 1:
                     # Read file in as little chunks
                     buf = f1.read(4096)
                     logger.debug("Type: {}".format(type(buf)))
                     if not buf : break
                     file_hash = hashlib.sha1(buf).digest()
                     logger.debug("File hash digest type: {}".format(type(file_hash)))
                     SHAhash.update(file_hash)
                f1.close()

    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        return -2

    return SHAhash.hexdigest()

def feature_hash(features):
    """Experimental"""
    return json.dumps(features, sort_keys=True)
