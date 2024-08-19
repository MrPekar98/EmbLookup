# Usage:
#        python3 main.py <CSV_DIR> <ALIAS_FILE> <INDEX_MAPPING_FILE> <PROCESSED_ALIASES> [-h] [-T <TIME LIMIT>]
#
# <CSV_FILE>: Directory of CSV files of tables to be linked to KG
# <ALIAS_FILE>: CSV file of KG entity aliases
# <INDEX_MAPPING_FILE>: Mapping CSV file of KG entities
# <PROCESSED_ALIASES>: CSV file of pre-processed alieases
# -h: Flag to tell that the tables have headers
# -T: Optional parameter to set a time limit in minutes after which the entity linking will stop

import pandas as pd
import torch

import utils
import dataset_helpers
import embedding_learner
import faiss_indexes
import sys
import csv
import os
import time

def get_kg_alias_data_loader(kg_alias_dataset_file_name, configs, dataset_name):
    batch_size = configs["embedding_model_configs"]["batch_size"]
    string_helper = dataset_helpers.StringDatasetHelper(configs, dataset_name)
    dataset = dataset_helpers.KGAliasDataset(kg_alias_dataset_file_name, string_helper)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False)
    return data_loader

def setup_and_train_model():
    kg_alias_dataset_file_name = sys.argv[4]
    output_file_name="emblookup.pth"
    dataset_name="dataset"

    utils.seed_random_generators(1234)
    configs = utils.load_configs()
    data_loader = get_kg_alias_data_loader(kg_alias_dataset_file_name, configs, dataset_name)

    emblookup_model = embedding_learner.train_embedding_model(configs, data_loader, dataset_name)
    utils.save_emblookup_model(emblookup_model, output_file_name)
    return emblookup_model

def index_kg_aliases():
    kg_alias_mapping_file_name = sys.argv[3]
    dataset_name="dataset"
    model_file_name="emblookup.pth"
    faiss_index_file_name="emblookup.findex"

    configs = utils.load_configs()
    data_loader = get_kg_alias_data_loader(kg_alias_mapping_file_name, configs, dataset_name)
    emblookup_model = utils.load_trained_emblookup_model(dataset_name, model_file_name)

    # The following is a memory inefficient approach as we convert all strings to embeddings in one shot
    # this is okay as both approximate and product quantized  faiss indexes need data to "train"
    # before indexing can be done
    # instead of sending a sample to train, we pass all the strings to train and index
    embeddings_list = []
    for step, (entity_index, alias_str_tensor, fasttext_embedding) in enumerate(data_loader):
        emblookup_embeddings = emblookup_model.get_embedding(alias_str_tensor, fasttext_embedding)
        embeddings_list.append(emblookup_embeddings)

    emblookup_embeddings = torch.vstack(embeddings_list)

    # Use the default arguments: 64 dimensions
    index = faiss_indexes.ApproximateProductQuantizedFAISSIndex()
    # Convert tensor to numpy as FAISS expects numpy
    index.add_embedding(emblookup_embeddings.numpy())
    index.save_index(faiss_index_file_name)

class LookupFromFAISSIndex:
    def __init__(self):
        self.dataset_name = "dataset"
        self.model_file_name="emblookup.pth"
        self.faiss_index_file_name="emblookup.findex"
        self.mapping_file_name=sys.argv[3]

        # Create an index class with default params
        self.index = faiss_indexes.ApproximateProductQuantizedFAISSIndex()
        self.index.load_index(self.faiss_index_file_name)
        self.emblookup_model = utils.load_trained_emblookup_model(self.dataset_name, self.model_file_name)

        configs = utils.load_configs()
        self.string_helper = dataset_helpers.StringDatasetHelper(configs, self.dataset_name)

        # Sometimes the alias/mention can be strings like null which Pandas will convert to np.nan
        # avoid this and read string as is
        df = pd.read_csv(self.mapping_file_name, keep_default_na=False, na_values=[''])
        self.mentions = df["Alias"].tolist()
        self.ids = df["KGID"].tolist()
        df = None
        self.index.set_index_to_mention_mapping(self.mentions)

        # load fasttext model with default parameters
        self.fasttext_model = utils.load_fasttext_model()

    def lookup(self, query):
        try:
            query = query.lower()
            alias_str_tensor = self.string_helper.string_to_tensor(query)
            alias_str_tensor = torch.unsqueeze(alias_str_tensor, dim=0)

            fasttext_embedding = torch.tensor(self.fasttext_model.get_word_vector(query))
            fasttext_embedding = torch.unsqueeze(fasttext_embedding, dim=0)

            embedding = self.emblookup_model.get_embedding(alias_str_tensor, fasttext_embedding)
            distances, indices, words = self.index.lookup(embedding.numpy(), k=5)

            return self.ids[indices[0][0]]

        except ValueError as e:
            return None

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('Missing argument. See top of this file for instructions.')
        exist(1)

    table_dir = sys.argv[1]
    output_file = 'results.csv'
    has_headers = len(sys.argv) > 5 and (sys.argv[5] == '-h' or sys.argv[6] == '-h' or sys.argv[7] == '-h')
    time_limit = -1

    if len(sys.argv) > 5:
        if sys.argv[5] == '-T':
            time_limit = int(sys.argv[6])

        elif sys.argv[6] == '-T':
            time_limit = int(sys.argv[7])

        else:
            print('Did not understand time limit parameter')
            exit(1)

    if not table_dir.endswith('/'):
        table_dir += '/'

    files = os.listdir(table_dir)

    print("Training EmbLookup model")
    emblookup_model = setup_and_train_model()

    print("Creating FAISS index based on embeddings")
    index_kg_aliases()

    print('Linking')
    emblookup = LookupFromFAISSIndex()
    linking_start = time.time()

    with open(output_file, 'w') as out_file:
        with open('runtimes.csv', 'w') as times_file:
            writer = csv.writer(out_file, delimiter = ',')
            time_writer = csv.writer(times_file, delimiter = ',')
            time_writer.writerow(['table', 'miliseconds'])

            for table_file in files:
                if time_limit > 0 and (time.time() - linking_start) / 60 >= time_limit:
                    print('Time limit reached')
                    break

                table_id = table_file.replace('.csv', '')

                with open(table_dir + table_file, 'r') as in_file:
                    row_i = 0
                    reader = csv.reader(in_file, delimiter = ',')
                    skip = has_headers
                    start = time.time() * 1000

                    for row in reader:
                        column_i = 0

                        if skip:
                            skip = False
                            continue

                        for column in row:
                            if not column.lstrip('-').replace('.', '', 1).replace('e-', '', 1).replace('e', '', 1).isdigit() and len(column) > 0:
                                entity = emblookup.lookup(column)

                                if not entity is None:
                                    writer.writerow([table_id, row_i, column_i, entity])

                            column_i += 1

                        row_i += 1

                    duration = time.time() * 1000 - start
                    time_writer.writerow([table_file, duration])
