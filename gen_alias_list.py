# Usage: python gen_alias_list.py <KG> <LABELS_TTL_FILE> <TTL_OUTPUT_FILE>
# <LABELS_TTL_FILE> can also be a directory, in which case this script will lookup for the alias predicate and use that for all files in the directory

import sys
import os

kg = sys.argv[1]
alias_predicate = '<http://dbpedia.org/ontology/alias>'

if kg == 'dbpedia':
    file = sys.argv[2]

    with open(file, 'r') as fd_in:
        with open(sys.argv[3], 'w') as fd_out:
            is_first = True

            for line in fd_in:
                if is_first:
                    is_first = False
                    continue

                elif '# ' in line and '2017' in line and ':' in line and '-' in line:
                    continue

                triple = line.split('>')
                subject = triple[0] + '>'
                object = triple[2][1:].replace('\\\"', '\'').replace('\'@en', '\"@en')
                fd_out.write(subject + ' ' + alias_predicate + ' ' + object)"""

elif kg == 'wikidata':
    dir = sys.argv[2]
    files = os.listdir(dir)

    with open(sys.argv[3], 'w') as out_file:
        i = 0

        for file in files:
            print(' ', end = '\r')
            print(str((i / len(files)) * 100)[:5] + '%', end = '\r')
            i += 1

            with open(dir + file, 'r') as in_file:
                for line in in_file:
                    if 'rdf-schema#label' in line:
                        triple = line.split('>')
                        subject = triple[0] + '>'
                        object = triple[2][1:].replace('\\\"', '\'').replace('\'@en', '\"@en')

                        if '@en' in object:
                            out_file.write(subject + ' ' + alias_predicate + ' ' + object)

else:
    print('KG \'' + kg + '\' not recognized')
