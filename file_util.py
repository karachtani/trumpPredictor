import os
import pandas as pd

def save_to_memory(dir_name, ticker, start_date, end_date, data):
    file_name = make_file_name(ticker, start_date, end_date)
    save_to_memory_with_fname(dir_name, file_name, data)

def save_to_memory_with_fname(dir_name, file_name, data):
    maybe_make_data_dir(dir_name)
    data.to_csv(dir_name + '/' + file_name)

def maybe_make_data_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_data_from_memory(dir_name, ticker, start_date, end_date):
    file_name = make_file_name(ticker, start_date, end_date)
    if os.path.isfile(dir_name + '/' + file_name):
        df = pd.read_csv(dir_name + '/' + file_name)
        return df
    else:
        return None

def make_file_name(ticker, start_date, end_date):
    return ticker + "_" + start_date.replace('-', '') + "_" + end_date.replace('-', '') + '.csv'