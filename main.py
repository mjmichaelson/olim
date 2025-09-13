
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

* output_dir: the dir to save in
* num_corpus_lines: the number of lines of the corpus to use, if null the whole corpus is used
* parallel: whether or not to use parallel processing -- generally for anything larger than 10,000 lines it probably makes sense
* overwrite: what to do if file exists

'''
def build_and_save_embeddings_to_file(output_dir='./embeddings', num_corpus_lines=10000, parallel=True, overwrite=False):
    import spacy
    import json
    import time
    import os

    if os.path.exists(output_dir):
        if not overwrite:
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
    print(f'Converting {num_corpus_lines} lines to SpaCy doc format (calculating embeddings)...')
    start_time = time.perf_counter()
    text = full_latin_corpus['train'][:num_corpus_lines]
    # parallelizable step:
    if parallel:
        text_doc = nlp.pipe(text, n_process=4)
    else:
        text_doc = [nlp.make_doc(chunk) for chunk in text]
    end_time = time.perf_counter()
    print(f'Conversion took {end_time - start_time} seconds')

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

def load_embedding_file(file_name):
    import spacy
    print(f'LOADING SAVED EMBEDDINGS')
    print('Loading latinCy model...')
    # when we load the model, we only want the Tok2Vec pipe in the spacy pipeline
    nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])
    print('LatinCy model loaded.')
    vocab = nlp.vocab

    # Deserialize the DocBin
    doc_bin = spacy.tokens.DocBin().from_file(file_name)

    # Get the individual Doc objects
    return list(doc_bin.get_docs(vocab))

def language_system_setup():
    # import dependencies
    import spacy
    
    # load latincy model into memory
    nlp = spacy.load("la_core_web_lg")

    # download text corpus



def main():
    language_system_setup()

if __name__ == '__main__':
    #main()

    build_and_save_embeddings_to_file(output_dir='./embeddings', num_corpus_lines=100000, parallel=True, overwrite=True)

    # get user target_phrase
    #target_phrase = 'sum magister' #'Urbis in Monte Tarpeio'
    #target_doc = nlp.make_doc(target_phrase)

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