'''
Additional utility functions for iBKH-KG preprocessing
'''

import os
import csv
import pandas as pd 
import numpy as np 
import pickle
from tqdm import tqdm
import glob

def generate_all_entity_csv(kg_folder="Data/iBKH/", ent_subdirectory="Entity", out_file="all_ent"):
    """
    Reads all **_vocab.csv files from the entity directory. Stores all entity names and their primary keys in a combined csv.
    """
    entity_folder = os.path.join(kg_folder, ent_subdirectory)
    entity_files = glob.glob(os.path.join(entity_folder, '*_vocab.csv'))

    all_ents = pd.concat([pd.read_csv(file) for file in entity_files])
    all_ents.to_csv(os.path.join(entity_folder, out_file + '.csv'), columns=['primary', 'name'], index=False)


def generate_id_label_maps(kg_folder="Data/iBKH/", ent_subdirectory="Entity", file="all_ent"):
    """
    Generates the following dicts and saves to pickle files:
    1. id2label: (primary key -> name)
    2. label2id: (name -> primary key)
    """
    entity_folder = os.path.join(kg_folder, ent_subdirectory)
    id2label = {}
    label2id = {}

    with open(os.path.join(entity_folder, file + ".csv"), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for id, name in reader:
            if id in id2label:
                continue
            id2label[id] = name
            label2id[name] = id

    id2labelfile = 'id2labeltest.pkl'
    label2idfile = 'label2idtest.pkl'

    with open(os.path.join(entity_folder, id2labelfile), 'wb') as file:
        pickle.dump(id2label, file)

    with open(os.path.join(entity_folder, label2idfile), 'wb') as file:
        pickle.dump(label2id, file)


def generate_triplet_head_tail_dict(triplet_path='Data/triplets/', 
                                    dict_path = 'Data/dicts/',
                                    triplet_filename = 'triplet_whole.csv',
                                    included_key_types = ["head", "tail", "headrel", "tailrel"],
                                    write_to_file = False
                                    ):
    """
    Generates the following dicts:
    1. head2tails_dict: maps head entities with list of tail entities (head -> list of tails)
    2. head2rels_dict: maps head entities with list of relations (head -> list of rels)
    3. tail2heads_dict: maps tail entities with list of head entities (tail -> list of heads)
    4. tail2rels_dict: maps tail entities with list of relations (tail -> list of rels)
    5. head_rel2tails_dict: maps head and relation combinations with list of tail entities (head@rel -> list of tails)
    6. tail_rel2heads_dict: maps tail and relation combinations with list of head entities (tail@rel -> list of heads)
    7. rel_tail2heads_dict: maps relation and tail combinations with list of tail entities (rel@tail -> list of heads)

    Args:
    - triplet_path (str): Filepath of csv with stored triples.
    - dict_path (str): Output directory.
    - triplet_filename (str): Filename of csv with stored triples.
    - included_key_types (list): types of dictionaries to generate. Valid values are ["head", "tail", "headrel", "tailrel"].
    - write_to_file (boolean): Option to write generated dicts to pickle files.

    Returns:
    - dict, dict, dict, dict, dict, dict, dict: generated dictionaries (None if not included).
    """
    try:
        triplet = pd.read_csv(triplet_path + triplet_filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Check filename: '{triplet_filename}' and path: '{triplet_path}'.")
    
    if write_to_file:
        os.makedirs(dict_path, exist_ok=True)
    
    head_list = list(triplet.drop_duplicates(subset='Head', keep='first')['Head'])
    tail_list = list(triplet.drop_duplicates(subset='Tail', keep='first')['Tail'])

    head2tails_dict, head2rels_dict = None, None 
    tail2heads_dict, tail2rels_dict = None 
    head_rel2tails_dict = None 
    tail_rel2heads_dict, rel_tail2heads_dict = None, None

    if "head" in included_key_types:
        # head -> list of tails
        # head -> list of rels
        head2tails_dict = {}
        head2rels_dict = {}
        for head_entity in tqdm(head_list, desc="Processing heads"):
            sub_df = triplet.loc[triplet['Head'] == head_entity]
            mapped_tail_list = list(sub_df.drop_duplicates(subset='Tail', keep='first')['Tail'])
            mapped_rel_list = list(sub_df.drop_duplicates(subset='Relation', keep='first')['Relation'])
            head2tails_dict[head_entity] = mapped_tail_list
            head2rels_dict[head_entity] = mapped_rel_list

        if write_to_file:
            with open(dict_path + 'head2tails_dict.pkl', 'wb') as f:
                pickle.dump(head2tails_dict, f)

            with open(dict_path + 'head2rels_dict.pkl', 'wb') as f:
                pickle.dump(head2rels_dict, f)

    if "tail" in included_key_types:
        # tail -> list of heads
        # tail -> list of rels
        tail2heads_dict = {}
        tail2rels_dict = {}
        for tail_entity in tqdm(tail_list, desc="Processing tails"):
            sub_df = triplet.loc[triplet['Tail'] == tail_entity]
            mapped_head_list = list(sub_df.drop_duplicates(subset='Head', keep='first')['Head'])
            mapped_rel_list = list(sub_df.drop_duplicates(subset='Relation', keep='first')['Relation'])
            tail2heads_dict[tail_entity] = mapped_head_list
            tail2rels_dict[tail_entity] = mapped_rel_list

        if write_to_file:
            with open(dict_path + 'tail2heads_dict.pkl', 'wb') as f:
                pickle.dump(tail2heads_dict, f)

            with open(dict_path + 'tail2rels_dict.pkl', 'wb') as f:
                pickle.dump(tail2rels_dict, f)

    if "headrel" in included_key_types:
        # head@rel -> list of tails
        head_rel2tails_dict = {}
        for (head, relation), group in tqdm(triplet.groupby(['Head', 'Relation']), desc="Processing head-rel groups"):
            key = f'{head}@{relation}'
            tail_list = list(group['Tail'].unique())
            head_rel2tails_dict[key] = tail_list

        if write_to_file:
            with open(dict_path + 'head@rel2tails_dict.pkl', 'wb') as f:
                pickle.dump(head_rel2tails_dict, f)

    if "tailrel" in included_key_types:
        # tail@rel -> list of heads
        # rel@tail -> list of heads
        tail_rel2heads_dict = {}
        rel_tail2heads_dict = {}
        for (tail, relation), group in tqdm(triplet.groupby(['Tail', 'Relation']), desc="Processing tail-rel groups"):
            key = f'{tail}@{relation}'
            key2 = f'{relation}@{tail}'
            head_list = list(group['Head'].unique())
            tail_rel2heads_dict[key] = head_list
            rel_tail2heads_dict[key2] = head_list

        if write_to_file:
            with open(dict_path + 'tail@rel2heads_dict.pkl', 'wb') as f:
                pickle.dump(tail_rel2heads_dict, f)

            with open(dict_path + 'rel@tail2heads_dict.pkl', 'wb') as f:
                pickle.dump(rel_tail2heads_dict, f)

    return head2tails_dict, head2rels_dict, tail2heads_dict, tail2rels_dict, head_rel2tails_dict, tail_rel2heads_dict, rel_tail2heads_dict


def examine_pkl(dict_path = 'Data/dicts/' ,filename='tail2heads_dict.pkl'):
    with open(dict_path + filename, 'rb') as file:
        data = pickle.load(file)

    if isinstance(data, dict):
        for i, (key, value) in enumerate(data.items()):
            print(f"{key}: {value}")
            if i == 4:
                break