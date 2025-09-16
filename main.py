
# Latin model courtesy of Patrick J. Burns, see https://spacy.io/universe/project/latincy and https://huggingface.co/latincy
# LatinCy paper here: https://arxiv.org/pdf/2305.04365v1

#!pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl

# Corpus is CC100 Latin https://arxiv.org/pdf/1911.02116 , Latin-only dataset here: https://huggingface.co/datasets/pstroe/cc100-latin/blob/main/README.md

#!curl -O -L https://huggingface.co/datasets/pstroe/cc100-latin/resolve/main/la.nolorem.tok.latalphabetonly.v2.json

# DEPRECATED Corpora courtesy of The Classical Language Toolkit (CLTK)
# https://github.com/cltk/tutorials/blob/master/2%20Import%20corpora.ipynb
# See https://github.com/cltk for all official corpora
#from cltk.data.fetch import FetchCorpus
#corpus_downloader = FetchCorpus(language="lat")
#corpus_downloader.list_corpora
#corpus_downloader.import_corpus("lat_text_latin_library")


'''
This reads in the Latin corpus file and converts it to a list of spaCy doc objects, assuming the default save=False.
If save=True, it creates a a serialized file of the spaCy 'doc' objects , saving it to disk.

Note that the corpus is fixed, as is the model used.
Only a tok2vec pipe is used because the only purpose of this file will be to support similarity calculations.

* num_corpus_lines: the number of lines of the corpus to use, if -1 the whole corpus is used
* n_process: number of procs to parallelize the conversion
* save: whether to save to disk or return result directly
* output_dir: the dir to save in
* overwrite: what to do if saving and file exists
'''
def build_embeddings(nlp=None, corpus=None, num_corpus_lines=100, n_process=1, save=False, 
                     output_dir='./embeddings', overwrite=False):
    import spacy
    import json
    import time
    import os

    if os.path.exists(output_dir):
        if save and not overwrite:
            print(f'Not building and saving embeddings. File at output dir already exists and \'overwrite\' is set to False')
            return

    print(f'BUILDING AND SAVING EMBEDDINGS')
    if not nlp:
        # load latincy model into memory
        # when we load the model, we only want the Tok2Vec pipe in the spacy pipeline
        print('No nlp model provided. Loading latinCy model...')
        nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])
        print('LatinCy model loaded.')

    if not corpus:
        print('Loading Latin corpus...')
        with open("../latin_text_data/la.nolorem.tok.latalphabetonly.v2.json") as f:
            corpus = json.load(f)
        print(f'Full corpus loaded as JSON -- train + test is {len(corpus)} lines long')

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

    if save:
        # collect the docs into one object for the write
        print(f'Collecting {num_corpus_lines} Docs into one DocBin...')
        print(f'/-',end='')
        start_time = time.perf_counter()
        doc_bin = spacy.tokens.DocBin()
        i=0
        for d in text_doc:
            doc_bin.add(d)
            if i % (num_corpus_lines//10) == 0:
                print('-',end='')
            i += 1
        end_time = time.perf_counter()
        print(f'/')
        print(f'Collection took {end_time - start_time} seconds')

        # Save the doc objects to the specified directory
        print(f'Writing DocBin to disk...')
        doc_bin.to_disk(output_dir)
        print(f"File of docs saved to: {output_dir}")
    else:
        return text_doc
    
'''
Options for saving:
    * if text list, then it saves as a .csv
    * if vector matrix, then it saves as .npy
    * if docbin, then it saves as docbin (.spacy)
'''
def save_corpus(embedding_generator, save_as='list', output_dir='./embeddings'):
    match save_as:
        case 'text list':
            import csv
            # create embedding matrix
            print(f'Creating text list...')
            start_time = time.perf_counter()
            corpus_list = list(embedding_generator)
            doc_text_list = [doc.text for doc in corpus_list]
            end_time = time.perf_counter()
            print(f'Creating text list took {end_time - start_time} seconds')
            print(f'Writing text list to disk...')
            with open("corpus_text_list.csv", "w", newline="") as file:
                writer = csv.writer(file)
                for i in doc_text_list:
                    writer.writerows([[i]])
        case 'vector matrix':
            # create embedding matrix
            print(f'Creating embedding matrix for calculation...')
            start_time = time.perf_counter()
            corpus_list = list(embedding_generator)
            doc_vectors = [doc.vector for doc in corpus_list]
            vector_matrix = np.stack(doc_vectors)
            end_time = time.perf_counter()
            print(f'Creating matrix took {end_time - start_time} seconds')
            print(f'Writing embedding matrix to disk...')
            np.save('corpus_embedding_matrix.npy', vector_matrix)
        case 'docbin':
            # collect the docs into one object for the write
            print(f'Collecting Docs into one DocBin...')
            print(f'/-',end='')
            start_time = time.perf_counter()
            doc_bin = spacy.tokens.DocBin()
            for d in embedding_generator:
                doc_bin.add(d)
            end_time = time.perf_counter()
            print(f'/')
            print(f'Collection took {end_time - start_time} seconds')

            # Save the doc objects to the specified directory
            print(f'Writing DocBin to disk...')
            doc_bin.to_disk(output_dir)
            print(f"File of docs saved to: {output_dir}")

def load_embeddings(file_path, type):
    if type == '.npy':
        res = np.load(file_path)
    elif type == '.csv':
        import csv
        res = []
        with open('corpus_text_list.csv', 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                res.append(row)
    elif type == 'docbin':
        res = load_docbin_embedding_file(file_path)
    return res


'''
This returns the contents of a byte file of spaCy DocBin(s) in that format
'''
def load_docbin_embedding_file(file_name):
    import spacy
    print(f'LOADING SAVED EMBEDDINGS')
    print('Loading latinCy model...')
    # when we load the model, we only want the Tok2Vec pipe in the spacy pipeline
    nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])
    print('LatinCy model loaded.')
    vocab = nlp.vocab

    # Deserialize the DocBin
    print('Deserializing saved file...')
    doc_bin = spacy.tokens.DocBin().from_disk(file_name)

    # return a list of docs
    return list(doc_bin.get_docs(vocab))


def calculate_spacy_similarity_scores_by_chunk(user_prompt_text, corpus_embeddings_chunk, nlp=None):
    # import dependencies
    import spacy
    
    if not nlp:
        # load latincy model into memory
        nlp = spacy.load("la_core_web_lg")
    
    # get embedding for user prompt
    user_prompt_text_doc = nlp.make_doc(user_prompt_text)

    # iterate over everything in the chunk to get similarities
    res_chunk = []
    for i, doc_i in enumerate(corpus_embeddings_chunk):
        res_chunk.append([i, doc_i.similarity(user_prompt_text_doc)])

    return res_chunk


def main():
    #language_system_setup()
    pass


if __name__ == '__main__':
    #main()

    import time
    import spacy
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])

    output_dir = './embeddings'

    # for my laptop, 4 procs concurrently is blazing fast, and fewer slows down considerably
    #corpus_with_embeddings = build_embeddings(nlp=nlp, corpus=None, num_corpus_lines=10000, n_process=4, 
    #                           save=False, output_dir=output_dir, overwrite=False)

    #save_corpus(corpus_with_embeddings, save_as='text list', output_dir='./embeddings')
    #save_corpus(corpus_with_embeddings, save_as='vector matrix', output_dir='./embeddings')

    corpus_list = load_embeddings(file_path='corpus_text_list.csv', type='.csv')
    vector_matrix = load_embeddings(file_path='corpus_embedding_matrix.npy', type='.npy')

    # create embedding matrix
    # print(f'Creating embedding matrix for calculation...')
    # start_time = time.perf_counter()
    # corpus_list = list(corpus_embeddings)
    # doc_vectors = [doc.vector for doc in corpus_list]
    # vector_matrix = np.stack(doc_vectors)
    # end_time = time.perf_counter()
    # print(f'Creating matrix took {end_time - start_time} seconds')

    # np.save('corpus_embedding_matrix.npy', vector_matrix)

    #loaded_matrix = np.load('corpus_embedding_matrix.npy')
    #print(loaded_matrix)

     # get user target_phrase
    target_phrase = 'sum magister' #'Urbis in Monte Tarpeio'
    target_doc = nlp.make_doc(target_phrase)

    # calculate distance from target to each row of corpus in one go, efficiently
    print(f'Calculating distances for {len(corpus_list)} rows...')
    start_time = time.perf_counter()
    corpus_similarities = cosine_similarity(target_doc.vector.reshape(1,-1), vector_matrix)
    end_time = time.perf_counter()
    print(f'Calculation took {end_time - start_time} seconds')

    # get sorted indices
    c2 = corpus_similarities.T
    sorted_indices = np.argsort(c2[:,0])[::-1]

    # display results
    for i in sorted_indices[:10]:
        print(c2[i], corpus_list[i])
