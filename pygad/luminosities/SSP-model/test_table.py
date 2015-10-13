#!/usr/bin/env python

import os
import numpy as np

print_diff = False

def read_table(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    head = ''.join(lines[1:11])
    names = lines[11][1:].split()
    table = np.loadtxt(filename)
    return head, names, table

directory = './'

if __name__ == '__main__':
    ref = read_table(directory + '/Z_0_0001.dat')
    all_fine = True
    for f in os.listdir(directory):
        if not f.startswith('Z_'):
            continue
        Z = float( f[2:8].replace('_','.') )
        h, n, t = read_table(directory + '/' + f)
        if h != ref[0]:
            print 'wrong header in:', f
            all_fine = False
            continue
        if n != ref[1]:
            print 'wrong table names in:', f
            all_fine = False
            continue
        if Z != t[0,0]:
            print 'wrong metallicity in:', f,
            print '   (Z = %f)' % t[0,0]
            all_fine = False
            continue
        if t.shape != ref[2].shape:
            print 'table has wrong shape in:', f,
            print '   (%s instead of %s)' % (t.shape, ref[2].shape)
            all_fine = False
            if t.shape[1] != ref[2].shape[1]:
                continue
        if print_diff:
            missing     = sorted( list(set(ref[2][:,1]) - set(t[:,1])) )
            additional  = sorted( list(set(t[:,1]) - set(ref[2][:,1])) )
            if additional:
                print 'additional values: ', \
                    '; '.join( map(lambda x: '%.5g'%x, additional) )
            if missing:
                print 'missing values:    ', \
                    '; '.join( map(lambda x: '%.5g'%x, missing) )

    if all_fine:
        print 'test passed'
