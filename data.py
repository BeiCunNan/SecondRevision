import json
import os
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, raw_data, label_dict):
        label_list = list(label_dict.keys())

        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_ids = label_dict[data['label']]
            dataset.append((tokens, label_ids))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


# Make tokens for every batch
def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=512,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)


# Load dataset
def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, workers,
              index_fold, subject):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'sst5':
        train_data = json.load(open(os.path.join(data_dir, 'SST5_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST5_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'terrible': 0, 'bad': 1, 'okay': 2, 'good': 3, 'perfect': 4}
    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'ie':
        train_data = json.load(open(os.path.join(data_dir, 'IE_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'IE_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1, 'other': 2}
    elif dataset == 'cr':
        data = json.load(open(os.path.join(data_dir, 'CR_CV.json'), 'r', encoding='utf-8'))
        oneFold_len = int(len(data) * 0.1)
        test_data = data[oneFold_len * index_fold:oneFold_len * index_fold + oneFold_len]
        train_data = data[:oneFold_len * index_fold] + data[oneFold_len * index_fold + oneFold_len:]
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        data = json.load(open(os.path.join(data_dir, 'SUBJ_CV.json'), 'r', encoding='utf-8'))
        oneFold_len = int(len(data) * 0.1)
        test_data = data[oneFold_len * index_fold:oneFold_len * index_fold + oneFold_len]
        train_data = data[:oneFold_len * index_fold] + data[oneFold_len * index_fold + oneFold_len:]
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'mr':
        data = json.load(open(os.path.join(data_dir, 'MR_CV.json'), 'r', encoding='utf-8'))
        oneFold_len = int(len(data) * 0.1)
        test_data = data[oneFold_len * index_fold:oneFold_len * index_fold + oneFold_len]
        train_data = data[:oneFold_len * index_fold] + data[oneFold_len * index_fold + oneFold_len:]
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'mpqa':
        data = json.load(open(os.path.join(data_dir, 'MPQA_CV.json'), 'r', encoding='utf-8'))
        oneFold_len = int(len(data) * 0.1)
        test_data = data[oneFold_len * index_fold:oneFold_len * index_fold + oneFold_len]
        train_data = data[:oneFold_len * index_fold] + data[oneFold_len * index_fold + oneFold_len:]
        label_dict = {'positive': 0, 'negative': 1}
    else:
        raise ValueError('unknown dataset')

    trainset = DataSet(train_data, label_dict)
    testset = DataSet(test_data, label_dict)

    collate_fn = partial(my_collate, tokenizer=tokenizer)
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn,
                                  pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn,
                                 pin_memory=True)
    return train_dataloader, test_dataloader
