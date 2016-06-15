import pandas as pd
import json
import os

def load_json_file(file_path, data_dict):
    with open('stack_exchange_data/corpus/apple/406.json') as f:
        for line in f:
            data_dict.append(json.loads(line))
    return data_dict

def load_json_as_pandas(file_path):
    '''
    Algorithm which reads in a json file or folder of files and transforms the data into a pandas dataframe
    Any non-json files will be skipped

    Args:
        file_path: File path to the json file, or folder containing the json files

    Returns:
        full_data_df: Pandas representation of the json file(s)
    '''

    full_data_dict = []
    #Check to see if a single json file is being passed in, and if so process only it
    if file_path.endswith('.json'):
        with open(file_path) as f:
                for line in f:
                    full_data_dict.append(json.loads(line))
    #Otherwise assume the user passed in a folder path, open each file in the folder and process
    else:
        try:
            for filename in os.listdir(file_path):
                if filename.endswith('.json'):
                    with open(os.path.join(file_path, filename)) as f:
                        for line in f:
                            full_data_dict.append(json.loads(line))
        except Exception as e:
            print("Only *.json files or folders can be processed.  Error processing: " + file_path)

            raise

    full_data_df = pd.DataFrame(full_data_dict)

    return full_data_df


