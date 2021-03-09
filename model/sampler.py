from taxo import TaxStruct
from torch.utils.data import Dataset as TorchDataset
import transformers
import torch
import numpy as np
import random


class Sampler:
    def __init__(self, tax_graph: TaxStruct):
        self._tax_graph = tax_graph
        self._nodes = list(self._tax_graph.nodes.keys())

    def sampling(self):
        margins = []
        pos_paths = []
        neg_paths = []
        for node, path in self._tax_graph.node2path.items():
            while True:
                neg_node = random.choice(self._nodes)
                if neg_node != path[0] and neg_node != self._tax_graph.root:
                    break
            pos_paths.append(path)
            neg_path = path[0:1] + self._tax_graph.node2path[neg_node]
            neg_paths.append(neg_path)
            margins.append(self.margin(path, neg_path))
        return pos_paths, neg_paths, margins

    @staticmethod
    def margin(path_a, path_b):
        com = len(set(path_a).intersection(set(path_b)))
        return min((abs(len(path_a) - com) + abs(len(path_b) - com)) / com, 3)


class Dataset(TorchDataset):
    def __init__(self, sampler: Sampler, tokenizer: transformers.BertTokenizer, word2des, padding_max=256,
                 margin_beta=0.1):
        self._sampler = sampler
        self._word2des = word2des
        self._padding_max = padding_max
        self._margin_beta = margin_beta
        self._tokenizer = tokenizer
        self._pos_paths, self._neg_paths, self._margins = self._sampler.sampling()

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos_path = self._pos_paths[item]
        neg_path = self._neg_paths[item]
        margin = self._margins[item]
        pos_ids, pos_type_ids, pos_attn_masks, pos_pool = self.encode_path(pos_path)
        neg_ids, neg_type_ids, neg_attn_masks, neg_pool = self.encode_path(neg_path)
        return dict(pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    pos_type_ids=pos_type_ids,
                    neg_type_ids=neg_type_ids,
                    pos_attn_masks=pos_attn_masks,
                    neg_attn_masks=neg_attn_masks,
                    pos_pool=pos_pool,
                    neg_pool=neg_pool,
                    margin=torch.FloatTensor([margin * self._margin_beta]))

    def encode_path(self, path):
        def_ids_l = [self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(w)) for w in path]
        pooling_matrix = [0.0] + [1.0 / float(len(def_ids_l[0]))] * len(def_ids_l[0])
        def_ids = []
        for def_id in def_ids_l:
            def_ids.extend(def_id)
            def_ids += [1]
        input_ids = [self._tokenizer.cls_token_id] + def_ids + [self._tokenizer.sep_token_id]
        input_len = len(input_ids)
        assert input_len <= self._padding_max
        input_ids = input_ids + [self._tokenizer.pad_token_id] * (self._padding_max - input_len)
        token_type_ids = [0] * self._padding_max
        attention_mask = [1] * input_len + [0] * (self._padding_max - input_len)
        pooling_matrix = pooling_matrix + [0] * (self._padding_max - len(pooling_matrix))
        return torch.LongTensor(input_ids), torch.LongTensor(token_type_ids), torch.LongTensor(
            attention_mask), torch.FloatTensor(pooling_matrix)
