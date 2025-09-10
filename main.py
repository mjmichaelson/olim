
# Latin model courtesy of Patrick J. Burns, see https://spacy.io/universe/project/latincy and https://huggingface.co/latincy
# LatinCy paper here: https://arxiv.org/pdf/2305.04365v1

#!pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-any-py3-none-any.whl

import spacy
nlp = spacy.load("la_core_web_lg")


def main():
    pass

if __name__ == '__main__':
    main()