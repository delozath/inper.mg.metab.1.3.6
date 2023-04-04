import os

import numpy  as np
import pandas as pd
import pdb


import ntxter
from ntxter.data.save      import to_excel, append_excel
from ntxter.data.loader    import load   as loader
from ntxter.data.datatools import common
from ntxter.data.explore   import identification

class ntxter_load:
    def __init__(self, path, bkname) -> None:
        self.cfg = loader.config(path, bkname)
        root_db = self.cfg['paths']['data']['root']
        folders = self.cfg['paths']['data']['folders']
        #
        def get_paths(f, r=root_db, os=os): return \
            (f[0], os.path
             .expanduser(
                f"~{(r + f[1]).replace('//', '/')}"
            ))
        #
        self.paths = dict(map(get_paths, folders.items()))
    #
    @classmethod
    def table_sheets(cls, path, bkname, db='db0'):
        inst = cls(path, bkname)
        #
        db_name = inst.cfg['block']['load']['tables'][db]
        sheets  = db_name.pop('sheets')
        var_sel = db_name.pop('var select')
        db_name = '.'.join(db_name.values())
        #
        inst.tables = loader.table(
            f"{inst.paths['dbs']}{db_name}", sheets=sheets)
        inst.var_sel = var_sel
        #
        return inst
    #
    def group_vars(self):
        self.data = self.tables['data']
        vars_  = self.tables['data_vars_sel' ].set_index('variable')
        vars_d = self.tables['data_var_descr'].set_index('variable')
        #
        #queries
        vars_  = vars_.query("analysis==@self.var_sel & vtype!='useless'")\
                      .drop(columns='analysis')\
                      .join(vars_d)
        #
        self.vars = vars_

class transform:
    def __init__(self, data, join, loader) -> None:
        self.data   = data
        self.join   = join
        self.loader = loader
    #
    def to_long_format(self):
        data = self.data.set_index('Expediente INPer')\
                        .drop(columns='Folio OBESO')
        #
        key  = 'cons'
        join = self.join.query("group==@key")
        cons = self._get_db_segment(key, data, join).drop(columns='time')
        #
        df = []
        for key in [0, 1, 3, 6]:
            join = self.join.query("group==@key")
            #df.append()
            tmp = self._get_db_segment(key, data, join)\
                      .join(cons)
            df.append(tmp)
        #
        df = pd.concat(df)#
        fname = self.loader.cfg['block']['save']['tables']['data']
        fname = '.'.join([*fname.values()])
        path  = self.loader.paths['dbs']
        append_excel(path + fname, {'long_format_db': df.reset_index()})
        print("File saved")
        pdb.set_trace()
        #    
    #
    def _get_db_segment(self, key, data, join):
        vnames = dict(
                    zip(*join[['variable', 'rename']]\
                            .to_dict(orient='list')\
                            .values()
                        )
                    )
        #
        tmp = data[[*vnames.keys()]].rename(columns=vnames)
        tmp['time'] = key
        #
        return tmp