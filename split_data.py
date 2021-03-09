import networkx as nx
import argparse
import codecs
import random


class TaxStruct(nx.DiGraph):
    def __init__(self, edges):
        super().__init__(edges)
        self.check_useless_edge()
        self._root = ""
        for node in self.nodes:
            if self.in_degree(node) == 0:
                self._root = node
                break
        assert self._root != ""

    def check_useless_edge(self):
        """
        删除对于taxonomy结构来说无用的边
        """
        bad_edges = []
        for node in self.nodes:
            if len(self.pred[node]) <= 1:
                continue
            # if self.out_degree(node) == 0:
            # print(node)
            for pre in self.predecessors(node):
                for ppre in self.predecessors(node):
                    if ppre != pre:
                        if nx.has_path(self, pre, ppre):
                            bad_edges.append((pre, node))
                            # print(node, pre, ppre)
        self.remove_edges_from(bad_edges)

    def all_leaf_nodes(self):
        # 根据是否只要单一父节点的叶节点可进行更改
        return [node for node in self.nodes.keys() if self.out_degree(node) == 0 and self.in_degree(node) == 1]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--taxo_path', type=str, help="path of full taxonomy dataset", required=False,
                            default="./data/raw_data/science_wordnet_en.taxo")
    arg_parser.add_argument('--train_path', type=str, help="output path of training dataset", required=False,
                            default="./data/train/science_train.taxo")
    arg_parser.add_argument('--terms_path', type=str, help="output path of terms file", required=False,
                            default="./data/eval/science_eval.terms")
    arg_parser.add_argument('--eval_path', type=str, help="output path of eval dataset", required=False,
                            default="./data/eval/science_eval.taxo")
    args, _ = arg_parser.parse_known_args()
    with codecs.open(args.taxo_path, encoding='utf-8') as f:
        # TAXONOMY FILE FORMAT: relation_id <TAB> term <TAB> hypernym
        tax_lines = f.readlines()
    tax_pairs = [[w for w in line.strip().split("\t")[1:]] for line in tax_lines]
    tax_pairs = [(p[1], p[0]) for p in tax_pairs]
    tax = TaxStruct(tax_pairs)
    leaf_nodes = tax.all_leaf_nodes()
    eval_terms = random.sample(leaf_nodes, int(len(tax.nodes) * 0.2))
    eval_lines = ["\t".join([term, list(tax.predecessors(term))[0]]) + "\n" for term in eval_terms]
    train_lines = ["\t".join(pair) + "\n" for pair in tax_pairs if pair[1] not in eval_terms]
    with codecs.open(args.terms_path, mode='w+', encoding='utf-8') as f:
        f.writelines([term + "\n" for term in eval_terms])
    with codecs.open(args.eval_path, mode='w+', encoding='utf-8') as f:
        f.writelines(eval_lines)
    with codecs.open(args.train_path, mode='w+', encoding='utf-8') as f:
        f.writelines(train_lines)
