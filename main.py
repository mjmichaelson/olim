
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

    # load text
    import spacy
    import json
    import time
    #from spacy.matcher import PhraseMatcher

    print('Loading latincy model...')
    # when we load the model, we only want the Tok2Vec pipe in the spacy pipeline
    nlp = spacy.load("la_core_web_lg", enable=["tok2vec"])
    print('Latincy model loaded.')

    print('Loading Latin corpus...')
    with open("../latin_text_data/la.nolorem.tok.latalphabetonly.v2.json") as f:
        full_latin_corpus = json.load(f)
    print(f'Full corpus loaded as JSON -- train + test is {len(full_latin_corpus)} lines long')

    
    # convert corpus data to proper format
    CORPUS_SIZE = 100000

    print(f'Converting {CORPUS_SIZE} lines to SpaCy doc format (calculating embeddings)...')
    start_time = time.perf_counter()
    text = full_latin_corpus['train'][:CORPUS_SIZE]
    # this is a parallelizable step:
    #text_doc = [nlp.make_doc(chunk) for chunk in text]
    text_doc = list(nlp.pipe(text, n_process=4))
    end_time = time.perf_counter()
    print(f'Conversion took {end_time - start_time} seconds')

    # get user target_phrase
    target_phrase = 'sum magister' #'Urbis in Monte Tarpeio'
    target_doc = nlp.make_doc(target_phrase)

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