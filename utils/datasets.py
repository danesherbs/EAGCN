from rdkit.Chem import AllChem as Chem
from eagcn_pytorch.neural_fp import fillBondType_dic
from eagcn_pytorch.neural_fp import fillAtomType_dic
import csv


def got_all_Type_solu_dic(dataset, path='../data/'):
    if dataset == 'Lipophilicity.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 2
        len_data = 4201
    elif dataset =='HIV.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 0
        len_data = 82255
    elif dataset == 'SAMPL.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 1
        len_data = 643
    elif dataset == 'tox21.csv':
        delimiter = ','
        quotechar = '"'
        smile_idx = 13
        len_data = 7832
    else:
        raise ValueError("Dataset {} not registered.".format(dataset))

    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            data.append(row)

    bondtype_dic = {}
    atomtype_dic = {}
    for row in data[1:len_data]:  # Wierd, the len(data) is longer, but no data was in the rest of part.
        if len(row) == 0:
            continue
        smile = row[smile_idx]
        try:
            mol = Chem.MolFromSmiles(smile)
            bondtype_dic = fillBondType_dic(mol, bondtype_dic)
            atomtype_dic = fillAtomType_dic(mol, atomtype_dic)
        except AttributeError:
            pass
        else:
            pass
    return(bondtype_dic, atomtype_dic)


def feature_normalize(x_all):
    """Min Max Feature Scalling for Atom Feature Matrix"""
    feature_num = x_all[0][0].shape[1]
    feature_min_dic = {}
    feature_max_dic = {}
    for i in range(len(x_all)):
        afm = x_all[i][0]
        afm_min = afm.min(0)
        afm_max = afm.max(0)
        for j in range(feature_num):
            if j not in feature_max_dic.keys():
                feature_max_dic[j] = afm_max[j]
                feature_min_dic[j] = afm_min[j]
            else:
                if feature_max_dic[j] < afm_max[j]:
                    feature_max_dic[j] = afm_max[j]
                if feature_min_dic[j] > afm_min[j]:
                    feature_min_dic[j] = afm_min[j]

    for i in range(len(x_all)):
        afm = x_all[i][0]
        feature_diff_dic = {}
        for j in range(feature_num):
            feature_diff_dic[j] = feature_max_dic[j]-feature_min_dic[j]
            if feature_diff_dic[j] ==0:
                feature_diff_dic[j] = 1
            afm[:,j] = (afm[:,j] - feature_min_dic[j])/(feature_diff_dic[j])
        x_all[i][0] = afm
    return x_all