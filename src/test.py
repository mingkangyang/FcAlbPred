from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import pickle
import h5py


with open(r'data/HSAonly_dict_4k.pkl', 'rb') as f: 
    ab_all_result_dict = pickle.load(f)

ab_name = list(ab_all_result_dict.keys())

aa_string = 'ARNDCEQGHILKMFPSTWYV'
char_to_int = {c: i+1 for i, c in enumerate(aa_string)}
non_standard_code = 0

def encode_seq_int(seq):
    return [char_to_int.get(c, non_standard_code) for c in seq]


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

    return positions, pre_idx, aft_idx



a_list, b_list, pos_list, y_list, mut_str_list = [], [], [], [], []


for pdb in ab_name:
    data = ab_all_result_dict[pdb]
    mut_strs = data['mutated_info']
    ddg = data['ddg']

    for mut_list, label in zip(mut_strs, ddg):
        positions, pre_idx, aft_idx = parse_mutation_string(mut_list, max_mut=5)

        a_list.append(pre_idx)
        b_list.append(aft_idx)
        pos_list.append(positions)
        y_list.append(label)

        mut_str_list.append(",".join(mut_list))


flat_a = np.array(a_list, dtype=np.int32)
flat_b = np.array(b_list, dtype=np.int32)
flat_pos = np.array(pos_list, dtype=np.int32)
flat_y = np.array(y_list, dtype=np.int32)


valid_mask = np.isin(flat_y, [1, 2])
flat_a, flat_b, flat_y = flat_a[valid_mask], flat_b[valid_mask], flat_y[valid_mask]
y_all = flat_y - 1  # 0/1


# valid_mask = np.isin(flat_y, [1, 2, 3])
# flat_a, flat_b, flat_y, flat_pos = (
#     flat_a[valid_mask],
#     flat_b[valid_mask],
#     flat_y[valid_mask],
#     flat_pos[valid_mask],
# )

# num_classes = 3
# y_all = keras.utils.to_categorical(flat_y - 1, num_classes=num_classes)
# print("y_all shape:", y_all.shape)



indices = np.arange(len(y_all))
train_val_idx, test_idx = train_test_split(indices, test_size=0.1, stratify=y_all, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.222, stratify=y_all[train_val_idx], random_state=42)


ckpt_path = "weight/HSA_FcRn_best_trainfit_4k_2class_aaproperties.weights.h5"
h5_path = 'data/binding_data_mut_2class_4k_aaproperties.h5'
batch_size = 8


def build_model(
    latent_dim=64,
    dense_units=64,
    max_mut=5,
    use_transformer=True,
    activation="sigmoid",
    # activation="softmax",
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
    output = layers.Dense(1, activation=activation)(x)
    # output = layers.Dense(3, activation=activation)(x)
    
    model = keras.Model(inputs=[Input_pre, Input_aft, Input_pos, Input_pre_prop, Input_aft_prop], outputs=output)
    return model

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
            



#### Plot ROC curve ###
model = build_model()
model.load_weights(ckpt_path)

test_gen = DataGenerator(test_idx, batch_size=batch_size, shuffle=False)

y_true = []
for i in range(len(test_gen)):
    _, y = test_gen[i]
    y = np.atleast_2d(y).T if y.ndim == 1 else y
    y_true.append(y)
y_true = np.vstack(y_true) # (N, 1) for binary, (N, 3) for multi-class

print("y_true shape:", y_true.shape)
print("Number of test samples:", y_true.shape[0])
print(y_true.flatten())


y_pred_prob = model.predict(test_gen)
print(y_pred_prob.flatten())

cm_to_inc = 1 / 2.54
width_cm = 4.5
height_cm = 3.0
figsize_custom = (width_cm * cm_to_inc, height_cm * cm_to_inc)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.linewidth'] = 1.0         

if y_true.ndim == 1 or y_true.shape[1] == 1:

    y_true = y_true.flatten().astype(int)
    y_pred_prob = y_pred_prob.flatten()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize_custom)
    ax = plt.gca()
    
    plt.plot(fpr, tpr, color='#1f77b4', lw=1.2, label=f'AUC={roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.8) #
    
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.legend(loc="lower right", fontsize=6, frameon=False, borderpad=0.2, handletextpad=0.4)
    
    ax.tick_params(axis='both', which='major', direction='in', 
                   width=1.0, length=3, labelsize=7, 
                   top=False, right=False)
    

    plt.savefig("binary_roc_curve_4k_aapro.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()

else:

    num_classes = y_true.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    plt.figure(figsize=figsize_custom)
    ax = plt.gca()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro ({roc_auc["micro"]:.2f})', 
             linestyle=':', lw=1.2, color='deeppink')
    plt.plot(all_fpr, mean_tpr, label=f'Macro ({roc_auc["macro"]:.2f})', 
             linestyle=':', lw=1.2, color='navy')

    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=0.8, color=colors[i%len(colors)], alpha=0.8,
                 label=f'C{i+1} ({roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
    
    plt.xlabel("")
    plt.ylabel("")
    plt.title("")
    
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.legend(loc="lower right", fontsize=4, frameon=False, 
               borderpad=0.1, labelspacing=0.1, handlelength=1.0)
    
    ax.tick_params(axis='both', which='major', direction='in', 
                   width=1.0, length=3, labelsize=7, top=False, right=False)
    
    plt.savefig("multiclass_roc_curve_5k_aapro.png", dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()