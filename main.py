#!/usr/bin/python3
# -*- coding : utf-8 -*-

import sys
import os
import pdb

SELF_PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
DSC_PATH  = os.path.expanduser('~/git/lib/')
sys.path.append(DSC_PATH)

import src

def main():
    load()
#
def load():
    block = 'block:p1'
    loader  = src.data.Load.table_sheets(SELF_PATH+'src/', block)
    loader.group_vars()
    loader.ptc_scrubbing_less()
    #data, vars_ = loader.tables['data'], loader.tables['data_vars_sel']
    model = src\
            .model_non_linear\
            .rbf_regression\
            .ft_sel_init(loader.data, 
                         loader.vars,
                         loader.cfg,
                         stage='pca')
    model.run()
    pdb.set_trace()


#
if __name__ == '__main__':
    # cd "../"
    # ./main.py
    main()
