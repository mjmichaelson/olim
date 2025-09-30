# olim

2025-09-25 v0.6 by Matt Michaelson

A tool to find what phrases in a corpus (eventually the classical corpus) most closely match your Latin phrase. As of v0.6, it runs on a large corpus of filtered latin web data in ~25 seconds per query (on my laptop) after corpus files are built.

### How does it work?

Olim uses a language model to determine a series of <a href="https://en.wikipedia.org/wiki/Embedding_(machine_learning)">embedding vectors</a> for the corpus, one per phrase, where each phrase embedding vector is the average of the individual token vectors that make up the phrase. Then, this matrix is used to calculate the <a href="https://en.wikipedia.org/wiki/Cosine_similarity">cosine similarity</a> between the user's input phrase and each phrase in the corpus. The results are sorted and the top 10 are displayed, along with their similarity scores.

### Installation

### Corpus
Corpus is CC100 Latin courtesy of Phillip Ströbel, see Latin-only dataset here: https://huggingface.co/datasets/pstroe/cc100-latin/blob/main/README.md
CC100 is based on https://arxiv.org/pdf/1911.02116 and ultimately derives from Common Crawl data.

### Model
Latin model courtesy of Patrick J. Burns, see https://spacy.io/universe/project/latincy and https://huggingface.co/latincy.
LatinCy paper here: https://arxiv.org/pdf/2305.04365v1

### Example Usage
Here are a few example queries using Olim v0.6:

INPUT: 'amor vincit omnia'

TIME: ~23 seconds

OUTPUT: 

| Match Score | Matched Phrase |
| -------- | -------- |
| [0.90429866] | Vincit omnia Veritas et amor vincit omnia | 
| [0.8986777] | omnia vincit amor virgilio omnia vincit amor et nos cedamus amori | 
| [0.8923784] | veritas et amor omnia vincit | 
| [0.8267149] | Omnia vincit amor et nos cedars amori | 
| [0.7858575] | omnia vincit amor et nos caedamus amori | 
| [0.783483] | Omnia vincit amor et nos cedamus amori | 
| [0.77088463] | omnia vincit amor et nos cedamus amori translatio -ne | 
| [0.7626245] | omnia vincit amor et nos cedamus amori keep calm and | 
| [0.7597208] | Amor mundum fecit , amor omnibus idem , amor patitur moras ! | 
| [0.757998] | Omnia vincit amor et nos cedamus amoriVergilius,Georgica | 
| [0.7508114] | Omnia vincit amor , et nos cedamus amori | 

From this is apparent several things about Olim results:
 * The data contains many phrases that are similar to each other
 * There is some metadata ('Vergilius,Georgica') unfortunately still mixed in
 * There is some English (!) mixed in, although it is rare ('keep calm and')
 * In practice it is difficult to confirm attribution of a phrase to an original document

### Ideas for improvement
Although the tool now works, it does not yet fulfill its core aim to look things up in the classical corpus. For a version 1.0:

* The corpus used should be a better-constructed and better-filtered corpus of all classical latin up to a certain year, say 200 CE, or perhaps, 'up to Augustine'. 
* It should be possible for the user to attribute matched phrases to their text of origin
* There should be a UI, even if it is only CLI