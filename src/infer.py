import os
import re
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, accuracy_score


aa_string = 'ARNDCEQGHILKMFPSTWYV'
char_to_int = {c: i+1 for i, c in enumerate(aa_string)}
non_standard_code = 0

def encode_seq_int(seq):
    return [char_to_int.get(c, non_standard_code) for c in seq]

aa_properties = {
    'A': {'hydro': 1.8, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 89.09},
    'R': {'hydro': -4.5, 'polar': 1, 'charge': 1, 'acid_base': 1,  'mol_wt': 174.20},
    'N': {'hydro': -3.5, 'polar': 1, 'charge': 0, 'acid_base': 0,  'mol_wt': 132.12},
    'D': {'hydro': -3.5, 'polar': 1, 'charge': -1,'acid_base': -1, 'mol_wt': 133.10},
    'C': {'hydro': 2.5, 'polar': 1, 'charge': 0, 'acid_base': 0,  'mol_wt': 121.16},
    'E': {'hydro': -3.5, 'polar': 1, 'charge': -1,'acid_base': -1, 'mol_wt': 147.13},
    'Q': {'hydro': -3.5, 'polar': 1, 'charge': 0, 'acid_base': 0,  'mol_wt': 146.15},
    'G': {'hydro': -0.4, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 75.07},
    'H': {'hydro': -3.2, 'polar': 1, 'charge': 1, 'acid_base': 1,  'mol_wt': 155.16},
    'I': {'hydro': 4.5, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 131.18},
    'L': {'hydro': 3.8, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 131.18},
    'K': {'hydro': -3.9, 'polar': 1, 'charge': 1, 'acid_base': 1,  'mol_wt': 146.19},
    'M': {'hydro': 1.9, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 149.21},
    'F': {'hydro': 2.8, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 165.19},
    'P': {'hydro': -1.6, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 115.13},
    'S': {'hydro': -0.8, 'polar': 1, 'charge': 0, 'acid_base': 0,  'mol_wt': 105.09},
    'T': {'hydro': -0.7, 'polar': 1, 'charge': 0, 'acid_base': 0,  'mol_wt': 119.12},
    'W': {'hydro': -0.9, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 204.23},
    'Y': {'hydro': -1.3, 'polar': 1, 'charge': 0, 'acid_base': 0,  'mol_wt': 181.19},
    'V': {'hydro': 4.2, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 117.15},
    'X': {'hydro': 0.0, 'polar': 0, 'charge': 0, 'acid_base': 0,  'mol_wt': 110.0}  # unknown/padding
}

from sklearn.preprocessing import StandardScaler


aa_df = pd.DataFrame(aa_properties).T

cols = ['hydro', 'polar', 'charge', 'acid_base', 'mol_wt']
aa_df = aa_df[cols]  
scaler = StandardScaler()
aa_df[['hydro', 'mol_wt']] = scaler.fit_transform(aa_df[['hydro', 'mol_wt']])
    
aa_prop_array = aa_df.to_dict(orient='index')

def encode_seq_properties(seq):

    features = []
    for aa in seq:
        prop = aa_prop_array.get(aa, aa_prop_array['X'])
        features.append(list(prop.values()))
    return np.array(features, dtype=np.float32)


def parse_mutation_string(mut_list, max_mut=5):
    """
    mut_list: ['D:V418M', 'D:T420A', 'D:V547A']
        positions: [418, 420, 547]
        pre_aas:   [V, T, V]
        aft_aas:   [M, A, A]
    """
    positions, pre_aas, aft_aas = [], [], []

    for m in mut_list[:max_mut]:
        if ':' in m:
            chain, rest = m.split(':')
        else:
            rest = m
        pre, pos, aft = rest[0], int(rest[1:-1]), rest[-1]
        positions.append(pos)
        pre_aas.append(pre)
        aft_aas.append(aft)

    # pad to max_mut
    while len(positions) < max_mut:
        positions.append(0)
        pre_aas.append('X')
        aft_aas.append('X')

    pre_idx = encode_seq_int(pre_aas)
    aft_idx = encode_seq_int(aft_aas)
    pre_prop = encode_seq_properties(pre_aas)
    aft_prop = encode_seq_properties(aft_aas)

    return positions, pre_idx, aft_idx, pre_prop, aft_prop

def infer_single(mut_list, model, max_mut=5):
    positions, pre_idx, aft_idx, pre_prop, aft_prop = parse_mutation_string(mut_list, max_mut=max_mut)
    X = [
        np.array([pre_idx], dtype=np.int32),
        np.array([aft_idx], dtype=np.int32),
        np.array([positions], dtype=np.int32),
        np.array([pre_prop], dtype=np.float32),  
        np.array([aft_prop], dtype=np.float32)   
    ]
    pred_prob = model.predict(X, verbose=0)
    num_classes = pred_prob.shape[1] if len(pred_prob.shape) > 1 else 1
    if num_classes == 1:
        prob = float(pred_prob[0, 0])
        pred_class = int(prob >= 0.5)
        return {"prob": prob, "pred_class": pred_class + 1}
    else:
        prob = pred_prob[0]
        pred_class = int(np.argmax(prob))
        return {"prob": prob.tolist(), "pred_class": pred_class + 1}

def build_model(latent_dim=64, dense_units=64, max_mut=5, use_transformer=True, num_classes=3, max_seq_len=600):
    Input_pre = keras.Input(shape=(max_mut,), dtype="int32", name="Input_pre")
    Input_aft = keras.Input(shape=(max_mut,), dtype="int32", name="Input_aft")
    Input_pos = keras.Input(shape=(max_mut,), dtype="int32", name="Input_pos")
    Input_pre_prop = keras.Input(shape=(max_mut, 5), dtype="float32", name="Input_pre_prop")##
    Input_aft_prop = keras.Input(shape=(max_mut, 5), dtype="float32", name="Input_aft_prop")##
    
    prop_proj = layers.Dense(latent_dim, activation="relu")##
    emb_layer = layers.Embedding(input_dim=21, output_dim=latent_dim, mask_zero=True)
    pos_layer = layers.Embedding(input_dim=max_seq_len + 1, output_dim=2 * latent_dim, mask_zero=True)
    
    pre_prop_emb = prop_proj(Input_pre_prop)##
    aft_prop_emb = prop_proj(Input_aft_prop)##
    emb_pre = emb_layer(Input_pre)
    emb_aft = emb_layer(Input_aft)
    emb_pos = pos_layer(Input_pos)
    
    emb_pre = layers.Concatenate()([emb_pre, pre_prop_emb])##
    emb_aft = layers.Concatenate()([emb_aft, aft_prop_emb])##
    
    diff = emb_aft - emb_pre + emb_pos
    diff = layers.Dense(latent_dim)(diff)##
    if use_transformer:
        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=latent_dim)(diff, diff)
        x = layers.LayerNormalization()(diff + attn_output)
        x = layers.GlobalMaxPooling1D()(x)
    else:
        x = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True))(diff)
        attn = layers.Attention()([x, x])
        x = layers.GlobalMaxPooling1D()(attn)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(dense_units // 2, activation="relu")(x)
    if num_classes == 2:
        output = layers.Dense(1, activation="sigmoid")(x)
    else:
        output = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=[Input_pre, Input_aft, Input_pos,Input_pre_prop,Input_aft_prop], outputs=output)
    return model

# def run_inference(ckpt_path, csv_path):

#     match = re.search(r"(\d+)class", ckpt_path)
#     num_classes = int(match.group(1)) if match else 2

#     model = build_model(num_classes=num_classes)
#     model.load_weights(ckpt_path)
#     df = pd.read_csv(csv_path)

#     results = []
#     for muts in df["Mutation"]:
#         mut_list = muts.strip('"').split(",")
#         res = infer_single(mut_list, model)
#         results.append({"Mutation": muts, "Prediction": res})

#     out_name = f"test_infer_results_{num_classes}class_87w_aaproperties.csv"
#     out_df = pd.DataFrame(results)
#     out_df.to_csv(out_name, index=False)



# from tensorflow.keras.utils import plot_model
# model = build_model()
# plot_model(model, to_file="mutation_model.png", show_shapes=True, show_layer_names=True)


# def run_inference_fast(ckpt_path, csv_path, batch_size=1024):
#     match = re.search(r"(\d+)class", ckpt_path)
#     num_classes = int(match.group(1)) if match else 2


#     model = build_model(num_classes=num_classes)
#     model.load_weights(ckpt_path)

#     df = pd.read_csv(csv_path)
#     all_pre_idx, all_aft_idx, all_pos = [], [], []
#     all_pre_prop, all_aft_prop = [], []

#     for muts in df["Mutation"]:
#         mut_list = muts.strip('"').split(",")
#         positions, pre_idx, aft_idx, pre_prop, aft_prop = parse_mutation_string(mut_list)
#         all_pre_idx.append(pre_idx)
#         all_aft_idx.append(aft_idx)
#         all_pos.append(positions)
#         all_pre_prop.append(pre_prop)
#         all_aft_prop.append(aft_prop)

#     X = [
#         np.array(all_pre_idx, dtype=np.int32),
#         np.array(all_aft_idx, dtype=np.int32),
#         np.array(all_pos, dtype=np.int32),
#         np.array(all_pre_prop, dtype=np.float32),
#         np.array(all_aft_prop, dtype=np.float32)
#     ]

#     pred_prob = model.predict(X, batch_size=batch_size, verbose=1)

#     results = []
#     if num_classes == 2:
#         probs = pred_prob.flatten()
#         preds = (probs >= 0.5).astype(int) + 1
#         for muts, p, c in zip(df["Mutation"], probs, preds):
#             results.append({"Mutation": muts, "prob": float(p), "pred_class": int(c)})
#     else:
#         preds = np.argmax(pred_prob, axis=1) + 1
#         for muts, p, c in zip(df["Mutation"], pred_prob.tolist(), preds):
#             results.append({"Mutation": muts, "prob": p, "pred_class": int(c)})

#     formatted = []
#     for r in results:
#         formatted.append({
#             "Mutation": r["Mutation"],
#             "Prediction": str({"prob": r["prob"], "pred_class": r["pred_class"]})
#         })

#     out_name = f"test_infer_results_{num_classes}class_test2_aaproperties_fast.csv"
#     out_df = pd.DataFrame(formatted)
#     out_df.to_csv(out_name, index=False, quoting=1)  

# csv_path = "data/test.csv"
# ckpt_paths = [
#     "Weight/HSA_FcRn_best_trainfit_1k_2class_aaproperties.weights.h5",
#     "Weight/HSA_FcRn_best_trainfit_3w_3class_aaproperties.weights.h5"
# ]

# for ckpt_path in ckpt_paths:
#     run_inference_fast(ckpt_path, csv_path, batch_size=1024)


def infer_prob_fast(model, df, batch_size=1024):
    all_pre_idx, all_aft_idx, all_pos = [], [], []
    all_pre_prop, all_aft_prop = [], []

    for muts in df["Mutation"]:
        mut_list = muts.strip('"').split(",")
        positions, pre_idx, aft_idx, pre_prop, aft_prop = parse_mutation_string(mut_list)
        all_pre_idx.append(pre_idx)
        all_aft_idx.append(aft_idx)
        all_pos.append(positions)
        all_pre_prop.append(pre_prop)
        all_aft_prop.append(aft_prop)

    X = [
        np.array(all_pre_idx, dtype=np.int32),
        np.array(all_aft_idx, dtype=np.int32),
        np.array(all_pos, dtype=np.int32),
        np.array(all_pre_prop, dtype=np.float32),
        np.array(all_aft_prop, dtype=np.float32),
    ]

    return model.predict(X, batch_size=batch_size, verbose=0)

from sklearn.metrics import roc_auc_score, accuracy_score

df = pd.read_csv(
    "data/test.csv"
)

y_true = (df["ddG(kcal/mol)"] == 1).astype(int).values


bin_model = build_model(num_classes=2)
bin_model.load_weights(
    "weight/HSA_FcRn_best_trainfit_1k_2class_aaproperties.weights.h5"
)

tri_model = build_model(num_classes=3)
tri_model.load_weights(
    "weight/HSA_FcRn_best_trainfit_3w_3class_aaproperties.weights.h5"
)


p_bin = infer_prob_fast(bin_model, df).flatten()
bin_high = 1.0 - p_bin   
print(f'bin_high={bin_high}')

p_tri = infer_prob_fast(tri_model, df)
tri_high = p_tri[:, 0] + p_tri[:, 1]
print(f'tri_high={tri_high}')

ws = np.linspace(0, 1, 101)
best_w, best_auc = None, -1

# for w in ws:
#     p_high = w * bin_high + (1 - w) * tri_high
#     auc = roc_auc_score(y_true, p_high)
#     print(f'w={w},auc={auc}')

#     if auc > best_auc:
#         best_auc = auc
#         best_w = w
###  w=0.48，ROC-AUC = 0.9870

p_high = 0.48 * bin_high + (1 - 0.48) * tri_high
# print(f'p_high={p_high}')
auc = roc_auc_score(y_true, p_high)
# print(f'auc={auc}')
