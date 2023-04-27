import pandas as pd
import os
import glob
import argparse

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--path', type=dir_path)
    return parser.parse_args()

def join_parquets_in_dir(path):
    df = pd.DataFrame()
    glob_path = ''.join([path,'/*.parquet'])
    for file in glob.glob(glob_path):
        print('Reading ... ' + file)
        pqt = pd.read_parquet(file)
        df = pd.concat([df,pqt])
    return(df)



def main():
    args = parse_arguments()
    out = join_parquets_in_dir(args.path)
    out_file = ''.join([args.path,'/merged.csv'])
    out.to_csv(out_file,index=None)
    print('Process Done')

if __name__ == "__main__":
    main()
