import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from rdkit import Chem
import networkx as nx
from utils import *

def get_atom_features(atom):
    return np.array(
        encode_one_hot_unknown(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 
                                                  'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 
                                                  'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 
                                                  'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        encode_one_hot(atom.GetDegree(), list(range(11))) +
        encode_one_hot_unknown(atom.GetTotalNumHs(), list(range(11))) +
        encode_one_hot_unknown(atom.GetImplicitValence(), list(range(11))) +
        [atom.GetIsAromatic()]
    )

def encode_one_hot(value, allowable_set):
    if value not in allowable_set:
        raise ValueError(f"Input {value} not in allowable set {allowable_set}")
    return [value == s for s in allowable_set]

def encode_one_hot_unknown(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return [value == s for s in allowable_set]

def convert_smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    atom_count = mol.GetNumAtoms()
    features = [get_atom_features(atom) / sum(get_atom_features(atom)) for atom in mol.GetAtoms()]

    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    graph = nx.Graph(edges).to_directed()
    edge_index = [[e1, e2] for e1, e2 in graph.edges]

    return atom_count, features, edge_index

def encode_sequence(protein):
    encoded_seq = np.zeros(max_sequence_length)
    for i, char in enumerate(protein[:max_sequence_length]):
        encoded_seq[i] = sequence_dict[char]
    return encoded_seq

def process_dataset(dataset_name):
    print(f'Converting data from DeepDTA for {dataset_name}')
    data_path = f'data/{dataset_name}/'
    train_indices = json.load(open(data_path + "folds/train_fold_setting1.txt"))
    train_indices = [idx for sublist in train_indices for idx in sublist]
    test_indices = json.load(open(data_path + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(data_path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(data_path + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinities = pickle.load(open(data_path + "Y", "rb"), encoding='latin1')

    if dataset_name == 'davis':
        affinities = [-np.log10(y / 1e9) for y in affinities]
    affinities = np.asarray(affinities)

    for phase in ['train', 'test']:
        rows, cols = np.where(~np.isnan(affinities))
        if phase == 'train':
            rows, cols = rows[train_indices], cols[train_indices]
        elif phase == 'test':
            rows, cols = rows[test_indices]

        with open(f'data/{dataset_name}_{phase}.csv', 'w') as file:
            file.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_index in range(len(rows)):
                compound_smile = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[rows[pair_index]]), isomericSmiles=True)
                target_sequence = proteins[cols[pair_index]]
                affinity_value = affinities[rows[pair_index], cols[pair_index]]
                file.write(f'{compound_smile},{target_sequence},{affinity_value}\n')

    print(f'\nDataset: {dataset_name}')
    print(f'Train fold size: {len(train_indices)}')
    print(f'Test fold size: {len(test_indices)}')
    print(f'Unique drugs: {len(set(ligands.values()))}, Unique proteins: {len(set(proteins.values()))}')

sequence_vocabulary = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
sequence_dict = {char: (i + 1) for i, char in enumerate(sequence_vocabulary)}
max_sequence_length = 1000

compound_smiles = set()
for dataset in ['kiba', 'davis']:
    for phase in ['train', 'test']:
        df = pd.read_csv(f'data/{dataset}_{phase}.csv')
        compound_smiles.update(df['compound_iso_smiles'])

smile_graphs = {smile: convert_smile_to_graph(smile) for smile in compound_smiles}

for dataset in ['davis', 'kiba']:
    train_file = f'data/processed/{dataset}_train.pt'
    test_file = f'data/processed/{dataset}_test.pt'
    if not (os.path.isfile(train_file) and os.path.isfile(test_file)):
        train_df = pd.read_csv(f'data/{dataset}_train.csv')
        train_drugs, train_proteins, train_affinities = train_df['compound_iso_smiles'], train_df['target_sequence'], train_df['affinity']
        train_proteins_encoded = [encode_sequence(protein) for protein in train_proteins]

        test_df = pd.read_csv(f'data/{dataset}_test.csv')
        test_drugs, test_proteins, test_affinities = test_df['compound_iso_smiles'], test_df['target_sequence'], test_df['affinity']
        test_proteins_encoded = [encode_sequence(protein) for protein in test_proteins]

        print(f'Preparing {dataset}_train.pt in PyTorch format!')
        train_data = TestbedDataset(root='data', dataset=f'{dataset}_train', xd=train_drugs, xt=train_proteins_encoded, y=train_affinities, smile_graph=smile_graphs)
        print(f'Preparing {dataset}_test.pt in PyTorch format!')
        test_data = TestbedDataset(root='data', dataset=f'{dataset}_test', xd=test_drugs, xt=test_proteins_encoded, y=test_affinities, smile_graph=smile_graphs)
        print(f'{train_file} and {test_file} have been created')
    else:
        print(f'{train_file} and {test_file} are already created')