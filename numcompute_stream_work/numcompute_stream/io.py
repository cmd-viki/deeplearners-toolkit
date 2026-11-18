"""NumPy-based CSV reader and bounded-memory chunk iterator."""
import csv
import numpy as np

def read_csv(path,skip_header=0,delimiter=','):
    data=np.genfromtxt(path,delimiter=delimiter,skip_header=skip_header,dtype=float,ndmin=2)
    if data.ndim!=2: raise ValueError('CSV must be 2D')
    return data

def iter_csv_chunks(path,chunk_size=100,skip_header=0,delimiter=','):
    if chunk_size<1: raise ValueError('chunk_size must be positive')
    with open(path,newline='',encoding='utf-8') as f:
        reader=csv.reader(f,delimiter=delimiter)
        for _ in range(skip_header): next(reader,None)
        chunk=[]
        for row in reader:
            if not row: continue
            chunk.append([float(v) if v.strip() else np.nan for v in row])
            if len(chunk)>=chunk_size:
                yield np.asarray(chunk,dtype=float); chunk=[]
        if chunk: yield np.asarray(chunk,dtype=float)
