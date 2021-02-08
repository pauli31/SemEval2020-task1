#!/usr/bin/env python3
"""
    reimplement in python the code from the CrossLingualSemantics group
    for canonical correlation transformation
    The original version of this code was in CCA.java and
    clss.transform.CanonicalCOrrelationTransformation.java
    That code mean-centers and unit-normalizes the embeddings before computing
    the transformation.   I don't know whether this causes problems,
    but based on our results, perhaps not.
    To compare results, I also did that.  
    I believe that Hardoon 
    and Artexte 
    and Press are relevant references for this version of the algorithm
"""
from gensim.models import KeyedVectors
import math
import numpy as np
#import random
import scipy.linalg as sl  #want to use scipy.linalg.svd w/ full_matrices=False
import sys

def transforms(x, y): #x and y are matrices made up of corresponding dict entries
    """
        x and y are two vectors of vectors with x[n] being an embedding vector
        The width of x and y can vary, and don't have to match.
        x[n] is the embedding vector for a word w_n in L_x, and 
        y[n] is the embedding vector for a word v_n in L_y, such that 
           w_n and v_n are corresponding words
        The goal of CCA is to return two linear transforms A and B, 
          such that
            x @ A == y @ B,
          and the transform function which calls this one returns
            R = A @ (inverse B),
        which transforms a vector in the x space to one in the y space
            L_x[w_n]@ R  = x[n] @ R = y[n] = L_y[v_n]
        The point of returning R is the hope that we can find approximate
        L_y entries for L_x entries which do not appear in x
                 
   The java version of this code uses a compact SVD, 
   whereas numpy.linalg.svd is a full svd, 
   this matters for 'tall matrices' like x and y,
   In the previous version, I truncate the output of nl.svd, but 
   here I 
   use a (hopefully faster) compact or truncating algorithm.
    """
    # get dimensions
    nx,mx = x.shape
    ny,my = y.shape
    typex = x.dtype

    if nx != ny or mx != my:
        raise Exception("The dimensions do not match")

    # factorize x, y with svd.  x = ux @ np.diag(s) @ vhx
    ux, sx, vhx = sl.svd(x, full_matrices = False) #scipy for compact feature
    uy, sy, vhy = sl.svd(y, full_matrices = False)

    #truncate ux, uy to n X m matrices
    ux = ux[:,:mx]
    uy = uy[:,:my]

    # turn the diagonal vectors into m by m matrices
    sx = np.diag(sx)
    sy = np.diag(sy)

    U1U2 = ux.T @ uy    # exciting variable name appears in java version of code

    up, sp, vhp = sl.svd(U1U2)

    # get inverses of sx and sy
    sxinv = sl.pinv(sx) 
    syinv = sl.pinv(sy)
    
    #pad inverses for multiply
    #Sxinv = np.zeros((x.shape[1],x.shape[0]) , dtype=np.float32)
    #Sxinv[:x.shape[1],:x.shape[1]] = sxinv
    #Syinv = np.zeros((y.shape[1],y.shape[0]) , dtype=np.float32)
    #Syinv[:y.shape[1],:y.shape[1]] = syinv

    # these are the matrices A and B mentioned in the top comment
    #    x @ A = y @ B
    A = vhx.T @ sxinv @ up
    B = vhy.T @ syinv @ vhp.T

    return A,B

def transform(X,Y):
    """
        call the Canonical Correlation Code above for the two transforms A,B`
        Then use  A and inverse of B to transform into the target space
    """
    A,B = transforms(X,Y)

    # compute the transform we promised.
    #  x @ A = y @ B

    binv = sl.pinv(B)
    R = A @ binv

    return R

def transform_KV(src,trg,target_words_dict, max_links=100000,
                deprecateBelow=(0,0),deprecateAbove=(10000,1.0)):
    """
        This function builds the translation dictionary for the
        diachronic comparison case, where src and trg have many words in
        common, and then 
        builds and returns the transform.

        src and trg are gensim KeyedVector models
        target_words_dict is a dictionary, whose keys are the target words
        max_links is the limit on number of translation items to use
        deprecateBelow is a pair = (border, fraction) where
            border is an index in the kv.vector table below which we are 
               fraction less likely to use the keys in the translation table
        deprecateAbove is a pair = (border, fraction) where
            border is an index in the kv.vector table above which we are 
               fraction less likely to use the keys in the translation table
    """
    # possibly add a tricky work-around.  Data structure for gensim <4.0.0 has no get_index():
    skeep = ensure_get_index(src)
    tkeep = ensure_get_index(trg)

    # decide on translation "pairs".
    # define generator for pairs
    def intersection():
        for x in src.vocab.keys():
            if not x.strip(): continue # if all white-space
            if x not in trg.vocab: continue
            if target_words_dict is not None and x in target_words_dict:
                continue

            # determine index class
            si = src.get_index(x)
            ti = trg.get_index(x)
            if si < deprecateBelow[0] or ti < deprecateBelow[0]:
                yield 0,x
            elif si > deprecateAbove[0] or si > deprecateAbove[0]:
                yield 2,x
            else:
                yield 1,x

    #dry run to count the translation dictonary possibles
    # keep track of three categories, depending index of vector:
    #  less than deprecateBelow  (probably a particle)
    #  greater than deprecateAbove (less common word)
    #  otherwise, "most well-supported differences"
    tdsiz = [0,0,0]
    for c,x in intersection():
        tdsiz[c] += 1


    # now insert randomly max_link of the possibles
    # it would be possible to eliminate tdict, directly build X and Y now
    # that we know their sizes... avoiding allocation would repay for dry run
    # but build_dict would have to be coded open.
    tdict = dict()
    tsum = tdsiz[0]*deprecateBelow[1]+tdsiz[1]+tdsiz[2]*deprecateAbove[1]
    slots_left = [int(max_links*tdsiz[0]/tsum*deprecateBelow[1]), 
                  int(max_links*tdsiz[1]/tsum), 
                  int(max_links*tdsiz[2]/tsum*deprecateAbove[1])]
    # want sum(slots_left) == max_links.  Adjust (probably up) "best" category
    slots_left[1] = max_links - slots_left[0]-slots_left[2] # sum to max_links
    pairs_left = tdsiz[0:]   #copy list
    for i,(c,x) in enumerate(intersection()):
        
        if (pairs_left[c] <= slots_left[c] or
            random.random()  <= slots_left[c]/pairs_left[c]):
                tdict[x] = x
                slots_left[c] += -1
        pairs_left[c] += -1

    print('debugging:',tdsiz, pairs_left, max_links, len(tdict))

    intersection = None # free space used by generator function

    # build translation arrays:
    X,Y = build_dict(tdict,src,trg)

    # and obtain the transform:
    trans_matrix = transform(X,Y)
    return trans_matrix

class junk:
    def __init__(self, mod):
        self.vocab = mod.vocab

    def get_index(self,key):
        """
            This function desiged to be inserted into a pre-4.0.0 gensim
            KeyedVectors model.  (Data structure was changed...)
        """
        return self.vocab[key].index


def ensure_get_index(KVmodel):
    """
        make sure that KVmodel has a get_index function, even if it is
        a pre-4.0.0 version of gensim model.
    """

    if 'get_index' in KVmodel.__dict__ : return KVmodel
    trf = junk(KVmodel)
    KVmodel.__dict__['get_index'] = trf.get_index
    return trf

def build_dict(dictionary, mx, my):
    """
        dictionary is either :
          a dict() object, 
            where dictionary.keys() are keys in mx
            and y=dictionary[x] is the corresponding key in my
                  or possibly None, if the values and keys are the same
           or 
          a file which is the name of a translation dictionary,
           which has two words per line,
            first has vector in Keyedvectors mx
            second has vector in Keyedvectors my
        mx and my are gensim KeyedVector models
        
        this code reads through the dictionary,
        taking a pair of words from each line.
        builds two arrays, X and Y, to be used in transform function
    """
    lx = []
    ly = []
    # I specialized this version for the DSC task; identical words
    if type(dictionary) == type({}):
        for x,y in dictionary.items():
            if y == None: y = x  # specialization for DSC
            lx.append(mx[x])
            ly.append(my[y])

    else: # dictionary is a file
      with open(dictionary) as fi:
        for lin in fi:
            line = lin.strip().split()
            if len(line) != 2 : continue
            x,y = line
            if not x in mx: continue
            if not y in my: continue
            lx.append(mx[x])
            ly.append(my[y])

    X = np.ndarray((len(lx),len(lx[0])), dtype = np.float32)
    Y = np.ndarray((len(ly),len(ly[0])), dtype = np.float32)
    for i,(x,y) in enumerate(zip(lx,ly)):
        X[i] = x
        Y[i] = y
    return X,Y

def cu_normalize(kv):
    """
    kv is a gensim KeyedVector object.  It supports a unit_normalization,
    but not a centering one, hence this routine
    normalization is a side-effect, so nothing is returned.
    I trade off  space for a little extra computation time with the for loop
    I could skip doing that calculation and call kv.init_sim()
    which creates an array of norms first:
        dist = sqrt((m**2)sum(-1))[...,np.newaxis]
        m = m/dist
    but that makes two passes over all the vectors, which probably don't
    fit in cache.  I already made two while mean-centering
    """
    origin = kv.vectors.sum(0) / kv.vectors.shape[0]
    kv.vectors = kv.vectors - origin
    """\
    for i,v in enumerate(kv.vectors):
        x = math.sqrt(v.dot(v))
        kv.vectors[i] = v/x
    """ # this code significantly slower than following borrowed from gensim
    dist = np.sqrt((kv.vectors ** 2).sum(-1))[..., np.newaxis]
    kv.vectors /= dist

def main1():
    """
    test timing on normalization
    """
    import timeit
    from gensim.models import KeyedVectors
    import math

    setups = """\
import math
import numpy as np
from numpy import newaxis
from gensim.models import KeyedVectors
emb = KeyedVectors.load_word2vec_format('scratch/fsw1', binary=False)
"""

    print('starting timing0',flush=True)
    t = timeit.timeit(setup=setups,
                  stmt ='dist = np.sqrt((emb.vectors ** 2).sum(-1))[..., newaxis];'+
                        'emb.vectors /= dist', number = 1000)
    code1 = """\
for i,v in enumerate(emb.vectors):
    x = math.sqrt(v.dot(v))
    emb.vectors[i] = v/x
"""
    code2 = """\
m = emb.vectors
dist = np.sqrt((m ** 2).sum(-1))[..., newaxis]
m /= dist
"""
    
    print(t)  # this result about 7.8 on my machine
    print('starting timing1',flush=True)
    t=timeit.timeit(setup=setups,
        stmt = code1, number =1000)
    
    print(t) # this result about 98 on my machine .  So loop loses.
    print('starting timing2',flush=True)
    t=timeit.timeit(setup=setups,
        stmt = code2, number =1000)

    print(t) # about 7.6 on my machine
    print('finished timing',flush=True)

def main():
    """
    Test the transform code.   To make sure I am matching the java outputs,
    I am mean-centering and L2-normalizing, but since the original
    embeddings do not contain these normalizations it seems possible that 
    the resulting transformation may be different than if I hadn't done them.
    At least the s values in the svd should be different...
    """
    n = 2
    fn = 'trans.dict'
    fx = 'fsw1' #None
    fy = 'fsw2' #None
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 4:
        fx = sys.argv[3]
        fy = sys.argv[4]

    # want to center and unit normalize the vectors
    mx = KeyedVectors.load_word2vec_format(fx,binary=False)
    my = KeyedVectors.load_word2vec_format(fy,binary=False)
    print("vectors loaded",file=sys.stderr,flush=True)
    
    cu_normalize(mx)
    cu_normalize(my)
    
    X,Y = build_dict(fn,mx,my)
    #Xs = X[:5001,:]   #this is a bug in the .java code, but i'll fix it
    #Ys = Y[:5001,:]   #only after I get the results to match
    print("starting build",file=sys.stderr,flush=True)
    R = transform(X,Y)
    for i in range(n):
        for j in range(n):
            print(R[i,j],end=' ')
        print()

if __name__ == '__main__': main()
