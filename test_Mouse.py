# import pyreadr
#
# result = pyreadr.read_r('data/Mouse embryo data/metadata.Rds') # also works for RData

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()

readRDS = robjects.r['readRDS']
df = readRDS('my_file.rds')
df = pandas2ri.ri2py(df)
