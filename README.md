# olim
2025-09-24 v0.6
Matt Michaelson

A tool to find what phrases in the classical corpus most closely match your Latin phrase.

As of v0.6, it runs on the whole corpus in ~25 seconds per query after corpus files are built.

### Installation

### Corpus
Corpus is CC100 Latin courtesy of Phillip Ströbel, see Latin-only dataset here: https://huggingface.co/datasets/pstroe/cc100-latin/blob/main/README.md
CC100 is based on https://arxiv.org/pdf/1911.02116.

### Model
Latin model courtesy of Patrick J. Burns, see https://spacy.io/universe/project/latincy and https://huggingface.co/latincy.
LatinCy paper here: https://arxiv.org/pdf/2305.04365v1

### Example Usage
Here are a few example queries using Olim v0.6:

INPUT: 'amor vincit omnia'
TIME: ~23 seconds
OUTPUT: 
Match Score // Matched Phrase
[0.90429866] Vincit omnia Veritas et amor vincit omnia
[0.8986777] omnia vincit amor virgilio omnia vincit amor et nos cedamus amori
[0.8923784] veritas et amor omnia vincit
[0.8267149] Omnia vincit amor et nos cedars amori
[0.7858575] omnia vincit amor et nos caedamus amori
[0.783483] Omnia vincit amor et nos cedamus amori
[0.77088463] omnia vincit amor et nos cedamus amori translatio -ne
[0.7626245] omnia vincit amor et nos cedamus amori keep calm and
[0.7597208] Amor mundum fecit , amor omnibus idem , amor patitur moras !
[0.757998] Omnia vincit amor et nos cedamus amoriVergilius,Georgica
[0.7508114] Omnia vincit amor , et nos cedamus amori

From this is apparent several things about Olim results:
 * The data contains many phrases that are similar to each other
 * There is some metadata ('Vergilius,Georgica') unfortunately still mixed in
 * There is some English (!) mixed in, although it is rare ('keep calm and')
 * In practice it is difficult to confirm attribution of a phrase to an original document