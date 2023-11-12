alias_predicate = '<http://dbpedia.org/ontology/alias>'
"""file = '../Tab2KG-benchmark/setup/tough_tables/dbpedia/labels_en.ttl'

with open(file, 'r') as fd_in:
    with open('aliases/alias_dbp10-2016.ttl', 'w') as fd_out:
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

import os

dir = '../Tab2KG-benchmark/setup/tough_tables/wikidata/'
files = os.listdir(dir)

with open('aliases/alias_wd.ttl', 'w') as out_file:
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
