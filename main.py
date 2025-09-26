'''
Olim, by Matt Michaelson

LATIN MODEL
Latin model courtesy of Patrick J. Burns, see https://spacy.io/universe/project/latincy and https://huggingface.co/latincy
LatinCy paper here: https://arxiv.org/pdf/2305.04365v1

!pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl

LATIN CORPUS
Corpus is CC100 Latin https://arxiv.org/pdf/1911.02116 , Latin-only dataset here: https://huggingface.co/datasets/pstroe/cc100-latin/blob/main/README.md

!curl -O -L https://huggingface.co/datasets/pstroe/cc100-latin/resolve/main/la.nolorem.tok.latalphabetonly.v2.json

DEPRECATED Previously used corpora courtesy of The Classical Language Toolkit (CLTK)
https://github.com/cltk/tutorials/blob/master/2%20Import%20corpora.ipynb
See https://github.com/cltk for all official corpora
from cltk.data.fetch import FetchCorpus
corpus_downloader = FetchCorpus(language="lat")
corpus_downloader.list_corpora
corpus_downloader.import_corpus("lat_text_latin_library")
'''


LATIN_CORPUS_JSON = "../latin_text_data/la.nolorem.tok.latalphabetonly.v2.json"
LATIN_CORPUS_LIST_FILENAME = 'corpus_text_list.csv'
LATIN_CORPUS_EMBEDDING_FILENAME = 'corpus_embedding_matrix.npy'
LATIN_CORPUS_EMBEDDING_MATRIX_SHAPE = (10366696, 300)


def build_corpus(nlp=None, num_corpus_lines=-1, overwrite=False):
    import os

    if overwrite:
        # build list
        build_list_file()
        # build matrix
        embedding_generator = build_embedding_generator(nlp=nlp, num_corpus_lines=num_corpus_lines, n_process=4)
        
        build_vector_matrix_file(filename=LATIN_CORPUS_EMBEDDING_FILENAME, doc_generator=embedding_generator, 
                                     corpus_length=LATIN_CORPUS_EMBEDDING_MATRIX_SHAPE[0])
    else:
        matrix_exists = os.path.exists(LATIN_CORPUS_EMBEDDING_FILENAME)
        list_exists = os.path.exists(LATIN_CORPUS_LIST_FILENAME)

        if not list_exists:
            # build list
            build_list_file()
        if not matrix_exists:
            # build matrix
            embedding_generator = build_embedding_generator(nlp=nlp, num_corpus_lines=num_corpus_lines, n_process=4)
            
            build_vector_matrix_file(filename=LATIN_CORPUS_EMBEDDING_FILENAME, doc_generator=embedding_generator, 
                                     corpus_length=LATIN_CORPUS_EMBEDDING_MATRIX_SHAPE[0])
    return


'''
This reads in the Latin corpus file and converts it to a generator of spaCy doc objects.

Note that the corpus is fixed, as is the model used.
Only a tok2vec pipe is used because the only purpose of this file will be to support similarity calculations.

* num_corpus_lines: the number of lines of the corpus to use, if -1 the whole corpus is used
* n_process: number of procs to parallelize the conversion
'''
def build_embedding_generator(nlp=None, num_corpus_lines=100, n_process=1):
    import spacy
    import json
    import time

    print(f'BUILDING EMBEDDING GENERATOR')
    if not nlp:
        # load latincy model into memory
        # when we load the model, we only want the Tok2Vec pipe in the spacy pipeline
        print('No nlp model provided. Loading latinCy model...')
        nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])
        print('LatinCy model loaded.')

    print('Loading Latin corpus...')
    with open(LATIN_CORPUS_JSON) as f:
        corpus = json.load(f)
    l = len(corpus['train']) + len(corpus['test'])
    print(f'Full corpus loaded as JSON -- train + test is {l} lines long')

    # convert corpus data to proper format
    l = 'all' if num_corpus_lines == -1 else num_corpus_lines
    print(f'Converting {l} lines to SpaCy doc format (calculating embeddings)...')
    start_time = time.perf_counter()
    if num_corpus_lines==-1:
        text = corpus['train']       
    else:
        text = corpus['train'][:num_corpus_lines]
    # parallelizable step:
    if n_process==1:
        text_doc = [nlp.make_doc(chunk) for chunk in text]        
    else:
        text_doc = nlp.pipe(text, n_process=n_process)
    end_time = time.perf_counter()
    print(f'Conversion took {end_time - start_time} seconds')

    return text_doc

def build_list_file(read_chunk_size=10000):
    import csv
    import pandas as pd

    json_reader = pd.read_json(LATIN_CORPUS_JSON, lines=True, chunksize=read_chunk_size)
    all_records = []
    for chunk in json_reader:
        all_records.extend(chunk.to_dict('records'))

    corpus_list = all_records[0]['train']
    corpus_list.extend(all_records[0]['test'])

    with open(LATIN_CORPUS_LIST_FILENAME, "w", newline="") as file:
        writer = csv.writer(file)
        for row in corpus_list:
            writer.writerow([row])

def read_text_list_file(file_path=LATIN_CORPUS_LIST_FILENAME):
    import csv
    res = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            res.append(row[0])
    return res

def build_vector_matrix_file(filename, doc_generator, corpus_length):
    shape = (corpus_length, LATIN_CORPUS_EMBEDDING_MATRIX_SHAPE[1])
    dtype = np.float32
    # create a new memory-mapped array
    mmap_array = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    # stream data one vector at a time
    i = 0
    for doc in doc_generator: 
        mmap_array[i:i+1] = doc.vector
        i += 1
    # write changes to disk
    mmap_array.flush()

def read_vector_matrix_file(filename=LATIN_CORPUS_EMBEDDING_FILENAME, shape=LATIN_CORPUS_EMBEDDING_MATRIX_SHAPE):
    return np.memmap(filename, dtype=np.float32, mode='r', shape=shape)

def slice_vector_matrix(mat, num_slices):
    if num_slices < 2:
        print('You must select at least 2 slices.')
        return
    res = []
    slice_l = len(mat)//num_slices
    start = 0
    end = slice_l
    while end <= len(mat):
        res.append(mat[start:end])
        start += slice_l
        end += slice_l
    return res


if __name__ == '__main__':

    import time
    import spacy
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])

    # if necessary, builds cachefiles of corpus data and embeddings
    build_corpus(nlp=nlp, num_corpus_lines=-1, overwrite=False)

    # now the files exist, so load
    corpus_list = read_text_list_file(file_path=LATIN_CORPUS_LIST_FILENAME)
    vector_matrix = read_vector_matrix_file(filename=LATIN_CORPUS_EMBEDDING_FILENAME,
                                            shape=LATIN_CORPUS_EMBEDDING_MATRIX_SHAPE) #(10366692, 300)

     # get user target_phrase
    target_phrase = 'nomen mihi est cloelia' 
    target_doc = nlp.make_doc(target_phrase)

    # calculate distance from target to each row of corpus without holding the whole embedding matrix in memory
    print(f'Calculating distances for {len(corpus_list)} rows...')
    start_time = time.perf_counter()
    sim_list = []
    for slice in slice_vector_matrix(vector_matrix, 12):
        sim_list.extend(cosine_similarity(target_doc.vector.reshape(1,-1), slice))
    corpus_similarities = np.concatenate(sim_list)
    end_time = time.perf_counter()
    print(f'Calculation took {end_time - start_time} seconds')

    # get sorted indices
    c2 = corpus_similarities.reshape(-1,1)
    sorted_indices = np.argsort(c2[:,0])[::-1]

    # display results
    for i in sorted_indices[:10]:
        print(c2[i], corpus_list[i])
