import pandas as pd
import numpy as np
import glob
import os
import argparse
import tqdm

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def parse_arguments():
    parser = argparse.ArgumentParser(description='The path of CSV files to process')
    parser.add_argument('--path', type=dir_path)
    return parser.parse_args()

def main():
	args = parse_arguments()
	file_list = glob.glob(''.join([args.path,'/*csv']))
	li = []
	for file in tqdm.tqdm(file_list):
	    df = pd.read_csv(file, index_col=None, header=0,low_memory=False)
	    li.append(df)
	print('concatenating ...')
	frame = pd.concat(li, axis=0, ignore_index=True)
	frame.to_csv('concat_data.csv',index=None)
	print('Done!')

if __name__ == '__main__':
	main()
