
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
This creates a serialized file of spaCy 'doc' objects for each line of the search_corpus, saving it to disk.
Only a tok2vec pipe is used because the only purpose of this file will be to support similarity calculations.

* num_corpus_lines: the number of lines of the corpus to use, if -1 the whole corpus is used
* n_process: number of procs to parallelize the conversion
* save: whether to save to disk or return result directly
* output_dir: the dir to save in
* overwrite: what to do if saving and file exists

'''
def build_embeddings(num_corpus_lines=10000, n_process=1, save=True, output_dir='./embeddings', overwrite=False):
    import spacy
    import json
    import time
    import os

    if os.path.exists(output_dir):
        if save and not overwrite:
            print(f'Not building and saving embeddings. File at output dir already exists and \'overwrite\' is set to False')
            return

    print(f'BUILDING AND SAVING EMBEDDINGS')
    print('Loading latinCy model...')
    # when we load the model, we only want the Tok2Vec pipe in the spacy pipeline
    nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])
    print('LatinCy model loaded.')

    print('Loading Latin corpus...')
    with open("../latin_text_data/la.nolorem.tok.latalphabetonly.v2.json") as f:
        full_latin_corpus = json.load(f)
    print(f'Full corpus loaded as JSON -- train + test is {len(full_latin_corpus)} lines long')

    # convert corpus data to proper format
    l = 'all' if num_corpus_lines == -1 else num_corpus_lines
    print(f'Converting {l} lines to SpaCy doc format (calculating embeddings)...')
    start_time = time.perf_counter()
    if num_corpus_lines==-1:
        text = full_latin_corpus['train']       
    else:
        text = full_latin_corpus['train'][:num_corpus_lines]
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


def load_embedding_file(file_name):
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

    # Get the individual Doc objects
    return list(doc_bin.get_docs(vocab))


def calculate_similarity_scores_by_chunk(user_prompt_text, corpus_embeddings_chunk, nlp=None):
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

    output_dir = './embeddings'

    text_doc = build_embeddings(num_corpus_lines=-1, n_process=4, save=False, output_dir=output_dir, overwrite=False)

    #corpus_embeddings = load_embedding_file(output_dir)

    #print(corpus_embeddings[:10])
    
    # get user target_phrase
    # target_phrase = 'sum magister' #'Urbis in Monte Tarpeio'
    # target_doc = nlp.make_doc(target_phrase)

    # # calculate similarity scores
    # print(f'Calculating similarity scores...')
    # start_time = time.perf_counter()
    # res = []
    # for i, doc_i in enumerate(text_doc):
    #     res.append([i, doc_i.similarity(target_doc)])
    # end_time = time.perf_counter()
    # print(f'Similarity scores took {end_time - start_time} seconds')

    # # sorted top 10 results
    # print('Sorting results...')
    # res = sorted(res, key=lambda elem: elem[1], reverse=True)[:10]

    # # find original top 10 phrases from corpus
    # print('Finding originals...')
    # for i, r in enumerate(res):
    #     res[i].append(text[res[i][0]])

    # for i in range(len(res)):
    #     print(f'{res[i]}')