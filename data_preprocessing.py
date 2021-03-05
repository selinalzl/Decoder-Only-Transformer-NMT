#!/usr/bin/env python3

"""
Preprocessing script to prepare the IWSLT dataset. Takes xml files and extracts 
source/target language sentences. The result files contain one sentence per line.

Code from https://pytorch.org/text/_modules/torchtext/datasets/translation.html
"""

import glob
import os
import codecs
import io
import xml.etree.ElementTree as ET


def clean(path):
    for f_xml in glob.iglob(os.path.join(path, '*.xml')):
        print(f_xml)
        f_txt = os.path.splitext(f_xml)[0]
        with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt:
            root = ET.parse(f_xml).getroot()[0]
            for doc in root.findall('doc'):
                for e in doc.findall('seg'):
                    fd_txt.write(e.text.strip() + '\n')

    xml_tags = ['<url', '<keywords', '<talkid', '<description',
                '<reviewer', '<translator', '<title', '<speaker']
    for f_orig in glob.iglob(os.path.join(path, 'train.tags*')):
        print(f_orig)
        f_txt = f_orig.replace('.tags', '')
        with codecs.open(f_txt, mode='w', encoding='utf-8') as fd_txt, \
                io.open(f_orig, mode='r', encoding='utf-8') as fd_orig:
            for line in fd_orig:
                if not any(tag in line for tag in xml_tags):
                    fd_txt.write(line.strip() + '\n')


if __name__ == "__main__":
    print("data preprocessing...")
    clean('iwslt/de-en/')
    print("\ndone")

