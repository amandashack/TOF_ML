#!/usr/bin/python3

import os
import numpy as np
import sandbox

global rng
rng = np.random.default_rng()
global nummax
nummax=20
global delta
delta=.025

def keepthis(h,ind):
    if (ind<0 or ind>(nummax-1)):
        return False
    if np.max(h)<2:
        return True
    if float(h[ind])/float(np.max(h) + delta) < rng.uniform(0,1):
        return True
    return False

def main():
    print(os.environ)
    if os.getenv('this'):
        print(os.getenv('this'))
    gaussdist=rng.normal(10,3,(1<<12))
    balancedist = []
    hist=np.zeros(nummax)
    newhist = np.zeros(nummax)
    inds = np.where((gaussdist>0) * (gaussdist<nummax))[0]
    for i in inds:
        hist[int(gaussdist[i])] += 1
    _=[print(' '*(int(v)>>3) + '.') for v in hist]
    if os.getenv('epochs'):
        _=[print('running silly epoch %i out of %s'%(i,os.environ.get('epochs'))) for i in range(int(os.getenv('epochs')))]
        sandbox.run_sandbox(os.getenv('epochs'))
    for v in gaussdist:
        if keepthis(newhist,int(v)):
            newhist[int(v)] += 1
            balancedist += [v]
    _=[print(' '*(int(v)) + '.') for v in newhist]
    return

if __name__ == '__main__':
    main()
