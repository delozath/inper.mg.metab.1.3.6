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
    proc = {'preproc to long format': prec_to_long_format,
            'test before quarto send': test_quarto}
    #
    steps = ['test before quarto send']
    for s in steps:
        print(f"Processing step [{s}]-->\n")
        proc[s]()
    #
    pdb.set_trace()

def test_quarto():
    block = 'block:lmm'
    loader  = src.data.ntxter_load.table_sheets(SELF_PATH, block)
    data    = loader.tables['long_format_db']
    pdb.set_trace()
#
def prec_to_long_format():
    block = 'block:1'
    loader  = src.data.ntxter_load.table_sheets(SELF_PATH, block)
    data    = loader.tables['Crecimiento']
    join    = loader.tables['join']
    #
    trans = src.data.transform(data, join, loader)
    trans.to_long_format()
#
if __name__ == '__main__':
    # cd "../"
    # ./main.py
    main()
