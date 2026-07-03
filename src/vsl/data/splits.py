import pandas as pd
import random
import json
import os

train_list = []
val_list = []
test_list = []

with open('merged\\front_view.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

for gloss in df['gloss'].unique():
    gloss_df = df[df['gloss'] == gloss]
    
    signers = list(gloss_df['signer_id'].unique())
    random.shuffle(signers)
    num_signers = len(signers)
    if num_signers < 3:
        train_list.append(gloss_df)
        continue  # Skip glosses with fewer than 3 signers

    train_end = max(1, int(0.7 * num_signers))
    val_size = max(1, int(0.15 * num_signers))
    val_end = train_end + val_size

    train_signers = signers[:train_end]
    val_signers = signers[train_end:val_end]
    test_signers = signers[val_end:]

    train_part = (gloss_df[gloss_df['signer_id'].isin(train_signers)])
    val_part = (gloss_df[gloss_df['signer_id'].isin(val_signers)])
    test_part = (gloss_df[gloss_df['signer_id'].isin(test_signers)])

    train_list.append(train_part)
    val_list.append(val_part)
    test_list.append(test_part)

train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True) 
test_df = pd.concat(test_list).reset_index(drop=True)


SPLIT_PATH = "split_data"
os.makedirs(SPLIT_PATH, exist_ok=True)

train_df.to_csv(os.path.join(SPLIT_PATH, "train.csv"), index=False)
val_df.to_csv(os.path.join(SPLIT_PATH, "val.csv"), index=False)
test_df.to_csv(os.path.join(SPLIT_PATH, "test.csv"), index=False)