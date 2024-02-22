# -*- coding: utf-8 -*-
"""preprocessing_NLG.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12SitMpGh9abCGkZNdXaSWpxWzxJpPE4d
"""

import numpy as np
import pandas as pd

from gensim.summarization.summarizer import summarize
from konlpy.tag import Komoran

import tarfile
import sys
import json
from tqdm import tqdm
import re

from sklearn.model_selection import train_test_split

import cleaning as cl
import eda

class data_preprosessing():
    """
    -----data preprocessing-----
    |   1. cleaning             |
    |   2. back translation     |
    |   3. data augmentation    |
    |___________________________|

    - settings:
        - module_path 본인 환경에 맞게 수정할 것(추후 업데이트 예정)
        - !pip install konlpy
        - !pip install pororo
        - !git clone -b yerim https://github.com/sw6820/STS.git

    - class : data_preprosessing(path, flag, size, n)
        param : 
            - path              : origin data file path
            - size=0.2          : train_valid split frac, default=0.2
            - n=4               : data augmentation times, default=4
            - random_state=None : split seed, default=None

    - functions : 
            - cleaning_df()         : input=None, output=DataFrame
            - back_translation(df)  : input=(cleaned)DataFrame, output=DataFrame
            - data_aug(df)          : input=(cleaned)DataFrame, output=DataFrame
            - load_all()            : input=None, output=DataFrame train, valid test

    - sample code :
        path = '[DATA PATH]'                                        # data 저장된 path
        size = 0.2                                                  # 0.1 - 1
        n = 4
        random_state=None
        pp = data_preprosessing(path, size, n, random_state)        # class param : path, size, n, random_state
        # _train, _valid, _test = pp.load_origin_to_split()         # optional
        train, valid, test = pp.load_all()                          # return augmented train, cleaned valid, cleaned test
    """


    def __init__(self, path, size=0.2, n=4, random_state=None):
        
        self.path = path                                                    # csv 파일 불러오고 저장할 path
        self.synonyms = pd.read_csv(path+'NIKLex_synonym.tsv', sep='\t')    # augmentation에서 사용할 말뭉치
        self.n = n                                                          # aumentation param
        self.size = size
        self.random_state = random_state
        self.ori_flie = path + 'sports_news_data.csv'
        self.col1 = 'title'
        self.col2 = 'content'
        self.bt_file = path + 'NLG_backtranslation.csv'

    def data_check(self, df, col1, col2):
        # 결측치 제거
        df = df.dropna(axis=0)
        
        # 중복값 제거
        df = df.drop_duplicates([col1,col2], keep='first')

        # BT data에서 train 값 뽑기 위한 index reset
        df = df.reset_index(drop=True)

        return df

    def load_origin_to_split(self):
        # original file load
        df = pd.read_csv(self.ori_flie)
        df = df.rename(columns={'TITLE':self.col1, 'CONTENT':self.col2})
        df = df.loc[:, self.col1:self.col2]
            
        # 결측치, 중복 제거
        df = self.data_check(df, self.col1, self.col2)

        # cleaning
        df = self.cleaning_df(df)            

        # label 생성
        print('\n**************make labels**************\n')
        df = self.make_labels(df)
            
        df = df.drop(columns='title', axis=1)

        self.col1 = 'content'
        self.col2 = 'summary'

        train, test = train_test_split(df, test_size=0.23, random_state=self.random_state)
        train, valid = train_test_split(train, test_size=self.size, random_state=self.random_state)

        valid = self.data_check(valid, self.col1, self.col2)
        test = self.data_check(test, self.col1, self.col2)

        print(f"\nLength of Cleaned DF : {len(df)}")
        print(f"\nLength of Cleaned train : {len(train)}")
        print(f"\nLength of Cleaned valid : {len(valid)}")
        print(f"\nLength of Cleaned test : {len(test)}")

        return train, valid, test

    def make_labels(self, df):
        df['summary']=None

        for i in tqdm(range(len(df))):
            try:
                length=0
                for j in df['content'][i].split('.')[:3]: # 첫 세 문장의 token을 lenght에 저장
                    length+=len(j.split(' '))

                s = summarize(df['content'][i], word_count=length)  # length만큼의 word를 선정하여 추출
                
                if len(s) < 1:
                    df['summary'][i] = np.nan
                else:
                    s = re.sub('\n', ' ', s)
                    df['summary'][i] = s
            except:
                df['summary'][i] = np.nan
        return df


    def cleaning_df(self , df):
        # data cleaning
        _cl = cl.DataCleaning()
        cl_df = _cl.make_cleaned_df(df, self.col1)
        
        return cl_df

    def back_translation(self, cl_df, lb_flag):

        _bt_df = pd.read_csv(self.bt_file)

        # 취합된 BT data에서 train index row 추출
        indexes = cl_df.index

        for i, index in enumerate(indexes):

            error_idx = []
            if i == 0:
                bt_df = _bt_df.iloc[[index]]
            else:
                try:
                    bt_df = bt_df.append(_bt_df.iloc[[index]])
                except:
                    error_idx.append(index)

        if lb_flag == True:
            print('\n**************make back translation labels**************\n')
            bt_df.drop(columns='summary', axis=1)
            bt_df = self.make_labels(bt_df)

        return bt_df

    def data_aug(self, cl_df, lb_flag):

        _eda = eda.NLPAugment(cl_df, self.synonyms)
        print(f'\ncolumn name : {self.col1}')
        da_df = _eda.augment_df_to_rows(self.col1, self.n)
        
        if lb_flag == True:
            print('\n**************make data augmentation labels**************\n')
            da_df.drop(columns='summary', axis=1)
            da_df = self.make_labels(da_df)

        return da_df

    def load_all(self, train=None, valid=None, test=None, lb_flag:bool=False, shuffle:bool=False):

        if train is not None:
            _train = train
            valid = valid
            test = test
        else:
            _train, valid, test = self.load_origin_to_split()

        print('\n**************cleaning : train, valid, test**************\n')
        cl_train = self.cleaning_df(_train)            # original train -> cleaning
        valid = self.cleaning_df(valid)                 # original test -> cleaning
        test = self.cleaning_df(test)                 # original test -> cleaning
        
        # data check null, duplicated
        cl_train = self.data_check(cl_train, self.col1, self.col2)
        valid = self.data_check(valid, self.col1, self.col2)
        test = self.data_check(test, self.col1, self.col2)
        print(f'\n**************cleaned train length : {len(cl_train)}, valid length : {len(valid)}, test length : {len(test)}**************\n')

        print('\n**************back translate : train**************\n')
        bt_df = self.back_translation(_train, lb_flag)    # train -> back translation
        bt_df = self.cleaning_df(bt_df)         # back translation -> cleaning
        bt_df = self.data_check(bt_df, self.col1, self.col2)    # data check null, duplicated
        print(f'\n**************back translate train length : {len(bt_df)}**************\n')

        print('\n**************data augmentation : train**************\n')
        da_df = self.data_aug(cl_train, lb_flag)  # cleaned data -> data augmentation
        da_df = self.cleaning_df(da_df)         # data augmentation -> cleaning
        da_df = self.data_check(da_df, self.col1, self.col2)    # data check null, duplicated
        print(f'\n**************data augmentation train length : {len(da_df)}**************\n')

        print('\n**************concat : train**************\n')
        train = pd.concat([cl_train, bt_df, da_df], axis=0, ignore_index=True)
        train = self.data_check(train, self.col1, self.col2)    # data check null, duplicated
        print(f'\n**************concat train length : {len(train)}**************\n')

        print(f'\n**************shuffle train, valid, test**************\n')
        if shuffle == True:
            train = train.sample(frac=1).reset_index(drop=True)
            valid = valid.sample(frac=1).reset_index(drop=True)
            test = test.sample(frac=1).reset_index(drop=True)

        print('\n**************saving files : train, valid, test**************\n')
        train.to_csv(self.path + 'train.csv', index=False)
        valid.to_csv(self.path + 'valid.csv', index=False)
        test.to_csv(self.path  + 'test.csv', index=False)

        print(f'\n**************train length : {len(train)}, valid length : {len(valid)}, test length : {len(test)}**************\n')
        return train, valid, test