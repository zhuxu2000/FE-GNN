import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from torch_geometric.transforms import BaseTransform
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'DU']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
                                           [Chem.rdchem.HybridizationType.SP,
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3,
                                            Chem.rdchem.HybridizationType.SP3D])
    return np.array(atom_features)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def process_smiles(df):
    train_y = np.array(df['y'])
    train_y = torch.tensor(train_y)
    datas = []
    mols = [Chem.MolFromSmiles(x) for x in df['smiles']]

    for mol, label in zip(mols, train_y):
        if mol is None:
            print("Invalid SMILES representation, unable to generate molecule, skipping this sample.")
            continue
        x = []
        for atom in mol.GetAtoms():
            x.append(get_atom_features(atom))
        x = torch.tensor(np.array(x), dtype=torch.float)
        edge_index = []
        edg_att = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge_index.append([start, end])
            edg_at = get_bond_features(bond)
            edg_att.append(edg_at)
        edg_att = torch.tensor(np.array(edg_att), dtype=torch.float)
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # 如果没有边，确保边索引为空
            edge_index = torch.empty((2, 0), dtype=torch.long)

            # 检查边索引和特征矩阵的大小
        if x.size(0) == 0:
            print("The graph lacks node features, skipping this sample.")
            continue
        if edge_index.size(1) == 0:
            print("The graph has no edges; the edge index is empty.")

        label = label.unsqueeze(0) if label.dim() == 0 else label
        data = Data(x=x,edge_attr=edg_att, edge_index=edge_index, y=label)
        datas.append(data)
    return datas

def process_multi(mols,lables):
    datas = []
    for mol, label in zip(mols, lables):
        if mol is None:
            print("Invalid SMILES representation, unable to generate molecule, skipping this sample.")
            continue
        x = []
        for atom in mol.GetAtoms():
            x.append(get_atom_features(atom))
        x = torch.tensor(np.array(x), dtype=torch.float)
        edge_index = []
        edg_att = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edge_index.append([start, end])
            edg_at = get_bond_features(bond)
            edg_att.append(edg_at)
        edg_att = torch.tensor(np.array(edg_att), dtype=torch.float)
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # 如果没有边，确保边索引为空
            edge_index = torch.empty((2, 0), dtype=torch.long)

            # 检查边索引和特征矩阵的大小
        if x.size(0) == 0:
            print("The graph lacks node features, skipping this sample.")
            continue
        if edge_index.size(1) == 0:
            print("The graph has no edges; the edge index is empty.")

        label = label.unsqueeze(0) if label.dim() == 0 else label
        data = Data(x=x, edge_attr=edg_att, edge_index=edge_index, y=label)
        datas.append(data)
    return datas

def train_test_load(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data,batch_size=64, shuffle=False)

    return train_loader,test_loader

def imp_data(datas, rf_factor=1):
    """
    对训练集和测试集数据进行处理，根据随机森林模型的特征重要性调整数据特征。

    :param train_set: train_data
    :param test_set: test_data
    :param rf_factor: 特征重要性调整因子，默认为1
    :return: 处理后的训练集和测试集数据
    """
    print('====================================================')
    print('M-score')
    train_loader,test_loader = train_test_load(datas)
    low_data = []
    lable = []
    for batch in train_loader:
        for i in range(len(batch)):
            single_x = batch.x[batch.ptr[i]:batch.ptr[i + 1]]
            single_x_sum = single_x.sum(dim=0, keepdim=True).squeeze().numpy()
            low_data.append(single_x_sum)
            single_y = batch.y[i].squeeze().numpy()
            lable.append(single_y)
    print(len(low_data))
    try:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(low_data, lable)
        importances = rf.feature_importances_
        importances = torch.tensor(importances, dtype=torch.float32)
        importances = importances / importances.mean()
        importances = importances + rf_factor
    except Exception as e:
        print(f"Error occurred during RandomForestRegressor fitting: {e}")
        raise

    def process_data(data_set, importances):
        #processed_data = []
        for data in data_set:
            try:
                data.x = data.x * importances[0]
            except Exception as e:
                print(f"Error occurred during data processing: {e}")
        return data_set

    train_data = process_data(train_loader, importances)
    test_data = process_data(test_loader, importances)
    return train_data, test_data

def process_ame(use_imp_data=None):
    df = pd.read_csv(r"dataset\Ames.smi", header=None, sep='\t')
    df.columns = ['smiles', 'CAS_NO', 'y']
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_bbbp(use_imp_data=None):
    df = pd.read_csv(r"dataset\BBBP.csv")
    df = df.rename(columns={'p_np': 'y'})
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_MUTAG(use_imp_data = None):
    dataset = TUDataset(root=r'dataset\TUDataset', name='MUTAG')
    data = dataset[0]  # Get the first graph object.
    print(data)
    print('=============================================================')
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'x: {data.x}')
    print(f'y: {data.y}')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(dataset)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(dataset)

    return train_loader, test_loader

def process_Bace(use_imp_data = None):
    df = pd.read_csv(r"dataset\bace.csv")
    df.rename(columns={'Class': 'y', 'mol': 'smiles'}, inplace=True)
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_clintox(use_imp_data = None):
    df = pd.read_csv(r'dataset\clintox.csv')
    df.rename(columns={'CT_TOX': 'y'}, inplace=True)
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_senolytic(use_imp_data = None):
    df = pd.read_csv(r"D:\FE-GNN\code\dataset\senolytic.csv")
    df.rename(columns={'SMILES': 'smiles'}, inplace=True)
    df.rename(columns={'senolytic': 'y'}, inplace=True)
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

class AddEdgeAttr(BaseTransform):
    def __init__(self, edge_feature_dim=6):
        self.edge_feature_dim = edge_feature_dim

    def __call__(self, data):
        num_edges = data.edge_index.size(1)
        edge_features = torch.zeros(num_edges, self.edge_feature_dim)
        data.edge_attr = edge_features
        return data

def process_NCI1(use_imp_data = None):
    # 应用自定义变换
    transform = AddEdgeAttr(edge_feature_dim=6)
    dataset = TUDataset(root=r'dataset\TUDataset', name='NCI1', transform=transform)
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    print(dataset[0])
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(dataset)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(dataset)

    return train_loader, test_loader

def process_NCI1_2(use_imp_data = None):
    # 应用自定义变换
    dataset = TUDataset(root=r'dataset\TUDataset', name='NCI1')
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    print(dataset[0])
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(dataset)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(dataset)

    return train_loader, test_loader


def process_esol(use_imp_data = None):
    df = pd.read_csv(r"dataset\esol.csv")
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_freesolve(use_imp_data = None):
    df = pd.read_csv(r"dataset\freesolv.csv")
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_HIV(use_imp_data = None):
    df = pd.read_csv(r"D:\FE-GNN\code\dataset\HIV.csv")
    df.rename(columns={'HIV_active': 'y'}, inplace=True)
    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    df = df.drop(none_list)
    print(df.head())
    data = process_smiles(df)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_tox21(use_imp_data = None):
    df = pd.read_csv(r"C:\Users\zx\Desktop\FE-GNN\code\dataset\tox21.csv")
    df = df.fillna(-1)
    label_columns = [col for col in df.columns if col not in ['smiles', 'mol_id']]
    label_ = np.array(df[label_columns])
    label_ = torch.tensor(label_)
    mols = [Chem.MolFromSmiles(x) for x in df['smiles']]
    data = process_multi(mols, label_)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_tox21(use_imp_data = None):
    df = pd.read_csv(r"D:\FE-GNN\code\dataset\tox21.csv")
    df = df.fillna(-1)
    label_columns = [col for col in df.columns if col not in ['smiles', 'mol_id']]
    label_ = np.array(df[label_columns])
    label_ = torch.tensor(label_)
    mols = [Chem.MolFromSmiles(x) for x in df['smiles']]
    data = process_multi(mols, label_)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def process_sider(use_imp_data = None):
    df = pd.read_csv(r"dataset\sider.csv")
    df = df.fillna(-1)
    label_columns = [col for col in df.columns if col not in ['smiles']]
    label_ = np.array(df[label_columns])
    print(label_.shape)
    label_ = torch.tensor(label_)
    mols = [Chem.MolFromSmiles(x) for x in df['smiles']]
    data = process_multi(mols, label_)
    if use_imp_data:
        print("======imp_data_process=====")
        train_loader, test_loader = imp_data(data)
    else:
        print("======data_process=====")
        train_loader, test_loader = train_test_load(data)

    return train_loader, test_loader

def load_dataset(dataset_name):
    if dataset_name == "NCI1":
        return process_NCI1(True)
    elif dataset_name == "Senolytic":
        return process_senolytic(True)
    elif dataset_name == "Bace":
        return process_Bace(True)
    elif dataset_name == "MUTAG":
        return process_MUTAG(False)
    elif dataset_name == "clintox":
        return process_clintox(False)
    elif dataset_name == "Ames":
        return process_ame(True)
    elif dataset_name == "BBBP":
        return process_bbbp(True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == '__main__':
    data_loader,test_loader = process_senolytic(False)
    data = next(iter(data_loader))
    print(data)
    x = data.x.numpy()
    x = x[:100]
    print(x.shape)
    plt.figure(figsize=(12, 8))
    sns.heatmap(x, cmap='viridis', cbar_kws={'label': 'values'})
    plt.tight_layout()
    plt.show()

