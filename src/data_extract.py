import pandas as pd
from biopandas.pdb import PandasPdb
import numpy as np
import pickle

def mutate_hsa_only(pdb_number, rbd_index_full):

    pdb_number_short = pdb_number[3:7] if pdb_number[0] == 'H' else pdb_number

    ab_data = pd.read_csv('data/HSA_mutation_input_3w.csv', encoding="ISO-8859-1")
    df_pdb = ab_data[ab_data['#PDB'] == pdb_number_short]
    mut_indexes = df_pdb['Mutation'].str.split(',')
    ddg = np.asarray(df_pdb['ddG(kcal/mol)'])

    ppdb = PandasPdb().read_pdb(f'data/{pdb_number_short}.pdb')
    residue_seq = ppdb.amino3to1()

    d_chain_mask = residue_seq['chain_id'] == 'D'
    residue_seq_d = residue_seq[d_chain_mask].copy()
    residue_seq_number_d = ppdb.df['ATOM']['residue_number'][residue_seq_d.index]

    pre_mut_seq = residue_seq_d['residue_name'].to_string(index=False).replace("\n", '')
    aft_mut_seq_list = []
    error = []
    k_list = []

    for k, mut_index in enumerate(mut_indexes):
        error_n = 0
        try:
            mut_residue_seq = residue_seq_d.copy()
            mut_residue_seq['residue_number'] = residue_seq_number_d
            for i in mut_index:
                mut_chain, mut_aa = i.split(':')
                if mut_chain != 'D':
                    continue  

                mut_pos = int(mut_aa[1:-1])
                pre_mut = mut_aa[0]
                aft_mut = mut_aa[-1]

                mut_subset_index = (mut_residue_seq['chain_id'] == mut_chain) & \
                                   (mut_residue_seq['residue_number'] == mut_pos)
                if not mut_subset_index.any():
                    raise ValueError(f"Residue not found: {mut_chain}{mut_pos}")

                mut_subset_index_pos = int(np.where(mut_subset_index)[0])

                if mut_residue_seq.iloc[mut_subset_index_pos]['residue_name'] == pre_mut:
                    mut_residue_seq.iat[mut_subset_index_pos, 1] = aft_mut
                else:
                    error.append(f'{pdb_number} aa {pre_mut} before mutation not match at chain {mut_chain} position {mut_pos}, deleting mutation at {k}')
                    k_list.append(k)
                    error_n = 1
                    break
        except (ValueError, IndexError, TypeError) as e:
            error.append(f'{pdb_number} delete mutation at {k}, error: {e}')
            k_list.append(k)
            error_n = 1

        if error_n == 0:
            aft_mut_seq = mut_residue_seq['residue_name'].to_string(index=False).replace("\n", '')
            aft_mut_seq_list.append([aft_mut_seq])

    ddg = np.delete(ddg, k_list)

    rbd_index_d = rbd_index_full[d_chain_mask.values]

    return {
        'pre_mut_seq': pre_mut_seq,
        'aft_mut_seq_list': aft_mut_seq_list,
        'mutated_info': mut_indexes,
        'ddg': ddg,
        'chain_index': residue_seq_d['chain_id'].to_string(index=False).replace("\n", ''),
        'residue_number': np.asarray(residue_seq_number_d),
        'alpha_carbon_coordinate': ppdb.df['ATOM'][['x_coord','y_coord','z_coord']].iloc[residue_seq_d.index].to_numpy(),
        'rbd_index': rbd_index_d,
        'error': error
    }

array = np.zeros((1, 946), dtype=int)
indices_to_set = [
    442, 443, 444, 445, 446, 447, 450, 470, 472, 777, 778, 779, 780, 782, 783,
    786, 821, 824, 825, 828, 829, 830, 858, 859, 860, 861, 862, 863, 864, 865,
    866, 867, 868, 869, 870, 871, 873, 885, 888, 889, 892, 895, 912, 934
]
rbd_index_full = array.reshape(-1)
rbd_index_full[indices_to_set] = 1

pdb_number = "4N0F"
ab_all_result_dict = {pdb_number: mutate_hsa_only(pdb_number, rbd_index_full)}

import pickle
with open('data/HSAonly_dict_3w.pkl', 'wb') as f:
    pickle.dump(ab_all_result_dict, f)

