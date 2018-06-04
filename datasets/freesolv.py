import csv
import operator
import numpy as np

from rdkit.Chem import AllChem as Chem

from utils.datasets import got_all_Type_solu_dic
from utils.datasets import feature_normalize


def load_data():
    return load_freesolv(path='../data/',
                         dataset='SAMPL.csv',
                         bondtype_freq=3,
                         atomtype_freq=3)


def load_freesolv(path='../data/', dataset = 'SAMPL.csv', bondtype_freq = 3,
                  atomtype_freq=3):
    print('Loading {} dataset...'.format(dataset))
    data = []
    with open('{}{}'.format(path, dataset), 'r') as data_fid:
        reader = csv.reader(data_fid, delimiter=',', quotechar='"')
        for row in reader:
            data.append(row)
    print('done')

    target = data[0][2]
    labels = []
    mol_sizes = []
    error_row = []
    bondtype_dic, atomtype_dic = got_all_Type_solu_dic(dataset)

    sorted_bondtype_dic = sorted(bondtype_dic.items(), key=operator.itemgetter(1))
    sorted_bondtype_dic.reverse()
    bondtype_list_order = [ele[0] for ele in sorted_bondtype_dic]
    bondtype_list_number = [ele[1] for ele in sorted_bondtype_dic]

    filted_bondtype_list_order = []
    for i in range(0, len(bondtype_list_order)):
        if bondtype_list_number[i] > bondtype_freq:
            filted_bondtype_list_order.append(bondtype_list_order[i])
    filted_bondtype_list_order.append('Others')

    sorted_atom_types_dic = sorted(atomtype_dic.items(), key=operator.itemgetter(1))
    sorted_atom_types_dic.reverse()
    atomtype_list_order = [ele[0] for ele in sorted_atom_types_dic]
    atomtype_list_number = [ele[1] for ele in sorted_atom_types_dic]

    filted_atomtype_list_order = []
    for i in range(0, len(atomtype_list_order)):
        if atomtype_list_number[i] > atomtype_freq:
            filted_atomtype_list_order.append(atomtype_list_order[i])
    filted_atomtype_list_order.append('Others')

    print('filted_atomtype_list_order: {}, \n filted_bondtype_list_order: {}'.format(filted_atomtype_list_order, filted_bondtype_list_order))

    x_all = []
    count_1 = 0
    count_2 = 0
    for i in range(1, len(data)):
        mol = Chem.MolFromSmiles(data[i][1])
        count_1 += 1
        try:
            (afm, adj, bft, adjTensor_OrderAtt,
             adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt) = Chem.molToGraph(
                mol,
                filted_bondtype_list_order,
                filted_atomtype_list_order).dump_as_matrices_Att()
            mol_sizes.append(adj.shape[0])
            labels.append([np.float32(data[i][2])])
            x_all.append([afm, adj, bft, adjTensor_OrderAtt, adjTensor_AromAtt, adjTensor_ConjAtt, adjTensor_RingAtt])
            count_2 +=1
        except AttributeError:
            print('the {}th row has an error'.format(i))
            error_row.append(i)
        except TypeError:
            print('the {}th row smile is: {}, can not convert to graph structure'.format(i, data[i][1]))
            error_row.append(i)
        i += 1

    x_all = feature_normalize(x_all)
    print('Done.')

    return(x_all, labels, target, mol_sizes)
