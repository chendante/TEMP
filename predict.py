import argparse
import model
import codecs
import torch
import transformers


class Predict:
    def __init__(self, args):
        self.args = args
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            # TAXONOMY FILE FORMAT: relation_id <TAB> term <TAB> hypernym
            tax_lines = f.readlines()
        tax_pairs = [[w for w in reversed(line.strip().split("\t")[1:])] for line in tax_lines]
        self.tax_graph = model.TaxStruct(tax_pairs)
        with codecs.open(args.terms, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.terms = [line.strip() for line in lines]
        self.model = model.TEMP.from_pretrained(args.model_path)
        self._tokenizer = transformers.BertTokenizer.from_pretrained(self.args.model_path)
        self.sampler = model.Sampler(self.tax_graph)
        self._padding_max = args.padding_max

    def predict(self):
        eval_group, tags = self.gen_eval_data()
        results = []
        eval_max = 1000
        for term in self.terms:
            data = eval_group[term]
            outputs = []
            data_l = int(data["ids"].size(0))
            for i in range(int((data_l - 1) / eval_max + 1)):
                begin = i * eval_max
                end = min((i + 1) * eval_max, data_l)
                with torch.no_grad():
                    output = self.model(input_ids=data["ids"][begin:end, ...].cuda(),
                                        token_type_ids=data["token_type_ids"][begin:end, ...].cuda(),
                                        attention_mask=data["attn_masks"][begin:end, ...].cuda(),
                                        pool_matrix=data["pool_matrices"][begin:end, ...].cuda())
                outputs.extend(output)
            outputs = torch.stack(outputs, dim=0)
            _, indices = outputs.squeeze().sort(descending=True)
            result = [tags[int(i)] for i in indices]
            results.append(result)
        return results

    def save_results(self, results):
        with codecs.open(self.args.output, mode='w+', encoding='utf-8') as fp:
            fp.write("\n".join(["\t".join(res) for res in results]))

    def gen_eval_data(self):
        path_group = dict()
        paths = []
        tags = []
        for k, v in self.tax_graph.node2path.items():
            tags.append(k)
            paths.append(v)
        for term in self.terms:
            ids_list = []
            token_type_ids_list = []
            attn_masks = []
            pool_matrices = []
            for path in paths:
                ids, token_type_ids, attn_mask, pool_matrix = self.encode_path([term] + path)
                ids_list.append(ids)
                token_type_ids_list.append(token_type_ids)
                attn_masks.append(attn_mask)
                pool_matrices.append(pool_matrix)
            path_group[term] = dict(ids=torch.stack(ids_list, dim=0),
                                    token_type_ids=torch.stack(token_type_ids_list, dim=0),
                                    attn_masks=torch.stack(attn_masks, dim=0),
                                    pool_matrices=torch.stack(pool_matrices, dim=0))
        return path_group, tags

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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--taxo_path', type=str, help="path of existing taxonomy dataset", required=False,
                            default="./data/raw_data/science_wordnet_en.taxo")
    arg_parser.add_argument('--model_path', type=str, help="path of trained model", required=False,
                            default="./data/model/")
    arg_parser.add_argument('--terms', type=str, help="path of prediction terms", required=False,
                            default="./data/eval/science_eval.terms")
    arg_parser.add_argument('--dict_path', type=str, help="path of dictionary file", required=False,
                            default="./data/")
    arg_parser.add_argument('--output', type=str, help="path of result file", required=False,
                            default="./data/result/science_result")
    arg_parser.add_argument('--padding_max', type=int, help="max num of tokens", required=False,
                            default=256)
    args, _ = arg_parser.parse_known_args()
    p = Predict(args)
    results = p.predict()
    p.save_results(results)
