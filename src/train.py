import sys
from biopandas.pdb import PandasPdb
import tensorflow as tf
import pandas as pd
import os
from collections import OrderedDict
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import time
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import pickle
import numpy as np
import string
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import h5py
import sys
print(sys.path)
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

if len(physical_devices) > 0:
    print(f"Have GPU: {physical_devices}")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
else:
    print("No GPU，use CPU")


with open(r'data/HSAonly_dict_3w.pkl', 'rb') as f: 
    ab_all_result_dict = pickle.load(f)

ab_name = list(ab_all_result_dict.keys())
print("PDBs in dict:", ab_name)
print("Top keys:", ab_name[:5])

for pdb in ab_all_result_dict:
    print(f"\n--- {pdb} ---")
    if isinstance(ab_all_result_dict[pdb], dict):
        print("Keys:", list(ab_all_result_dict[pdb].keys()))
    else:
        print("Type:", type(ab_all_result_dict[pdb]))
    break

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
aa_df = aa_df[cols]  # 
scaler = StandardScaler()
aa_df[['hydro', 'mol_wt']] = scaler.fit_transform(aa_df[['hydro', 'mol_wt']])
    
aa_prop_array = aa_df.to_dict(orient='index')

def encode_seq_properties(seq):

    features = []
    for aa in seq:
        prop = aa_prop_array.get(aa, aa_prop_array['X'])
        features.append(list(prop.values()))
    return np.array(features, dtype=np.float32)

##################define model######################

def build_model(
    latent_dim=64,
    dense_units=64,
    max_mut=5,
    use_transformer=True,
    # activation="sigmoid",
    activation="softmax",
    max_seq_len=600  
):

    Input_pre = keras.Input(shape=(max_mut,), dtype="int32", name="Input_pre")
    Input_aft = keras.Input(shape=(max_mut,), dtype="int32", name="Input_aft")
    
    Input_pos = keras.Input(shape=(max_mut,), dtype="int32", name="Input_pos")
    
    Input_pre_prop = keras.Input(shape=(max_mut, 5), dtype="float32", name="Input_pre_prop")
    Input_aft_prop = keras.Input(shape=(max_mut, 5), dtype="float32", name="Input_aft_prop")
    
    emb_layer = layers.Embedding(input_dim=21, output_dim=latent_dim, mask_zero=True)
    pos_layer = layers.Embedding(input_dim=max_seq_len+1, output_dim=2 * latent_dim, mask_zero=True)
    
    prop_proj = layers.Dense(latent_dim, activation="relu")

    pre_prop_emb = prop_proj(Input_pre_prop)
    aft_prop_emb = prop_proj(Input_aft_prop)

    emb_pre = emb_layer(Input_pre)
    emb_aft = emb_layer(Input_aft)
    emb_pos = pos_layer(Input_pos)
    
    emb_pre = layers.Concatenate()([emb_pre, pre_prop_emb])
    emb_aft = layers.Concatenate()([emb_aft, aft_prop_emb])

    diff = emb_aft - emb_pre + emb_pos  # (B, max_mut, latent_dim)
    diff = layers.Dense(latent_dim)(diff) 
    if use_transformer:
        # Transformer block
        attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=latent_dim)(diff, diff)
        x = layers.LayerNormalization()(diff + attn_output)
        x = layers.GlobalMaxPooling1D()(x)
    else:
        # BiLSTM + Attention
        x = layers.Bidirectional(layers.LSTM(latent_dim, return_sequences=True))(diff)
        attn = layers.Attention()([x, x])
        x = layers.GlobalMaxPooling1D()(attn)
    
    # Dense head
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(dense_units // 2, activation="relu")(x)
    # output = layers.Dense(1, activation=activation)(x)
    output = layers.Dense(3, activation=activation)(x)
    
    model = keras.Model(inputs=[Input_pre, Input_aft, Input_pos, Input_pre_prop, Input_aft_prop], outputs=output)
    return model

def parse_mutation_string(mut_list, max_mut=5):
    """
    mut_list: ['D:V418M', 'D:T420A', 'D:V547A']
    return：
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



a_list, b_list, pos_list, y_list = [], [], [], []
a_prop_list, b_prop_list = [], []


for pdb in ab_name:
    data = ab_all_result_dict[pdb]
    mut_strs = data['mutated_info']
    ddg = data['ddg']

    for mut_list, label in zip(mut_strs, ddg):
        positions, pre_idx, aft_idx, pre_prop, aft_prop = parse_mutation_string(mut_list, max_mut=5)
        a_list.append(pre_idx)
        b_list.append(aft_idx)
        pos_list.append(positions)
        a_prop_list.append(pre_prop)
        b_prop_list.append(aft_prop)
        y_list.append(label)

flat_a = np.array(a_list, dtype=np.int32)
flat_b = np.array(b_list, dtype=np.int32)
flat_pos = np.array(pos_list, dtype=np.int32)
flat_a_prop = np.array(a_prop_list, dtype=np.float32)
flat_b_prop = np.array(b_prop_list, dtype=np.float32)
flat_y = np.array(y_list, dtype=np.int32)


###binary classification: only keep ddG in {1,2} ###
# valid_mask = np.isin(flat_y, [1, 2])
# flat_a, flat_b, flat_y = flat_a[valid_mask], flat_b[valid_mask], flat_y[valid_mask]
# y_all = flat_y - 1  # 0/1

### 3-class classification: only keep ddG in {1,2,3} ###
valid_mask = np.isin(flat_y, [1, 2, 3])
flat_a, flat_b, flat_y, flat_pos = (
    flat_a[valid_mask],
    flat_b[valid_mask],
    flat_y[valid_mask],
    flat_pos[valid_mask],
)
num_classes = 3
y_all = keras.utils.to_categorical(flat_y - 1, num_classes=num_classes)
print("y_all shape:", y_all.shape)

with h5py.File("data/binding_data_mut_3class_5w_aaproperties.h5", "w") as f:
    f.create_dataset("flat_a", data=flat_a, compression="gzip")
    f.create_dataset("flat_b", data=flat_b, compression="gzip")
    f.create_dataset("flat_pos", data=flat_pos, compression="gzip")
    f.create_dataset("flat_a_prop", data=flat_a_prop, compression="gzip")
    f.create_dataset("flat_b_prop", data=flat_b_prop, compression="gzip")
    f.create_dataset("y", data=y_all, compression="gzip")

indices = np.arange(len(y_all))
train_val_idx, test_idx = train_test_split(indices, test_size=0.1, stratify=y_all, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.222, stratify=y_all[train_val_idx], random_state=42)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")


h5_path = 'data/binding_data_mut_3class_5w_aaproperties.h5'
class DataGenerator(keras.utils.Sequence):
    def __init__(self, indices, batch_size=8, shuffle=True):
        self.indices = np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_path = h5_path

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    
    def __getitem__(self, idx):

        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]

        if len(batch_indices) == 0:
            raise ValueError("Empty batch indices")

        if not np.all(np.diff(batch_indices) >= 0):  
            sorted_order = np.argsort(batch_indices)
            reverse_order = np.argsort(sorted_order)  
            sorted_batch_indices = batch_indices[sorted_order]

            with h5py.File(self.file_path, 'r') as f:

                flat_a_data = f['flat_a'][sorted_batch_indices]
                flat_b_data = f['flat_b'][sorted_batch_indices]
                flat_pos_data = f['flat_pos'][sorted_batch_indices]
                flat_a_prop_data = f['flat_a_prop'][sorted_batch_indices]
                flat_b_prop_data = f['flat_b_prop'][sorted_batch_indices]
                y_data = f['y'][sorted_batch_indices]

                X_batch = [
                    flat_a_data[reverse_order],
                    flat_b_data[reverse_order],
                    flat_pos_data[reverse_order],
                    flat_a_prop_data[reverse_order],
                    flat_b_prop_data[reverse_order]
                ]
                y_batch = y_data[reverse_order].astype(np.float32)  # 0/1
        else:

            with h5py.File(self.file_path, 'r') as f:
                X_batch = [
                    f['flat_a'][batch_indices],
                    f['flat_b'][batch_indices],
                    f['flat_pos'][batch_indices],
                    f['flat_a_prop'][batch_indices],
                    f['flat_b_prop'][batch_indices]
                ]
                y_batch = f['y'][batch_indices].astype(np.float32)  # 0/1
                # y_batch = keras.utils.to_categorical(y_batch, self.num_classes)
                
        return tuple(X_batch), y_batch      
    
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

class SaveBestValAcc(keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.best_val_acc = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy') 
        if val_acc is not None and val_acc > self.best_val_acc:
            print(f"\nValidation accuracy improved from {self.best_val_acc:.4f} to {val_acc:.4f}, saving model.")
            self.best_val_acc = val_acc
            self.model.save_weights(self.save_path)  
            

### 3-class classification ---
model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

###  binary classification ---
# model = build_model()
# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
#     loss="binary_crossentropy",
#     metrics=["accuracy", keras.metrics.AUC()]  
# )

batch_size = 12
train_gen = DataGenerator(train_idx, batch_size=batch_size, shuffle=True)
val_gen   = DataGenerator(val_idx, batch_size=batch_size, shuffle=False)

# classes = np.unique(y_all[train_idx])
# weights = compute_class_weight('balanced', classes=classes, y=y_all[train_idx])
# class_weight = dict(enumerate(weights))
# print("Class weights:", class_weight)

ckpt_path = "HSA_FcRn_best_trainfit_3w_3class_aaproperties.weights.h5"
save_best_val_acc_cb = SaveBestValAcc(ckpt_path)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[save_best_val_acc_cb],
    verbose=1
)









#####Plot training curves#####

hist = history.history

####Loss curve####
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

cm_to_inc = 1 / 2.54
width_cm = 3.5
height_cm = 3.0
figsize_small = (width_cm * cm_to_inc, height_cm * cm_to_inc)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 0.8  

plt.figure(figsize=figsize_small)
ax = plt.gca()

plt.plot(hist["loss"], label="Train", color='#1f77b4', lw=1.0)
plt.plot(hist["val_loss"], label="Val", color='#ff7f0e', lw=1.0)

plt.xlabel("")
plt.ylabel("")
plt.title("")

ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

ax.tick_params(axis='both', which='major', direction='in', 
               width=0.8, length=2, labelsize=6,
               top=False, right=False)

plt.legend(fontsize=5, frameon=False, loc='best', borderpad=0.1)

plt.savefig("loss_curve_3w_3class.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()


####Accuracy curve####
plt.figure(figsize=figsize_small)
ax = plt.gca()

plt.plot(hist["accuracy"], label="Train", color='#2ca02c', lw=1.0)
plt.plot(hist["val_accuracy"], label="Val", color='#d62728', lw=1.0)

plt.xlabel("")
plt.ylabel("")
plt.title("")

ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

ax.tick_params(axis='both', which='major', direction='in', 
               width=0.8, length=2, labelsize=6,
               top=False, right=False)

plt.legend(fontsize=5, frameon=False, loc='best', borderpad=0.1)

plt.savefig("accuracy_curve_3w_3class.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()
