import codecs
import argparse
import model


def score(results, tax_graph, trues):
    def get_wu_p(node_a, node_b):
        # if node_a == node_b:
        #     return 1.0
        full_path_a = node2full_path[node_a]
        full_path_b = node2full_path[node_b]
        com = full_path_a.intersection(full_path_b)
        lca_dep = 0
        for node in com:
            if len(tax_graph.node2path[node]) > lca_dep:
                lca_dep = len(tax_graph.node2path[node])
        dep_a = len(tax_graph.node2path[node_a])
        dep_b = len(tax_graph.node2path[node_b])
        res = 2.0 * float(lca_dep) / float(dep_a + dep_b)
        # assert res <= 1
        return res

    node2full_path = tax_graph.get_node2full_path()
    wu_p, acc, mrr = 0.0, 0.0, 0.0
    for result, ground_true in zip(results, trues):
        if result[0] == ground_true:
            acc += 1
        num = 0
        for i, r in enumerate(result):
            if r == ground_true:
                num = i + 1.0
                break
        mrr += 1.0 / num
        wu_p += get_wu_p(result[0], ground_true)
    acc /= float(len(results))
    mrr /= float(len(results))
    wu_p /= float(len(results))

    return acc, mrr, wu_p


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--results_path', type=str, help="output path of result file", required=False,
                            default="./data/result/science.results")
    arg_parser.add_argument('--eval_path', type=str, help="output path of eval dataset", required=False,
                            default="./data/eval/science_eval.taxo")
    arg_parser.add_argument('--taxo_path', type=str, help="path of existing taxonomy", required=False,
                            default="./data/train/science_train.taxo")
    args, _ = arg_parser.parse_known_args()
    with codecs.open(args.results_path) as fp:
        lines = fp.readlines()
    results = [line.strip().split("\t") for line in lines]
    with codecs.open(args.eval_path) as fp:
        lines = fp.readlines()
    trues = [line.strip() for line in lines]
    with codecs.open(args.taxo_path, encoding='utf-8') as f:
        # TAXONOMY FILE FORMAT: hypernym <TAB> term
        tax_lines = f.readlines()
    tax_pairs = [line.strip().split("\t") for line in tax_lines]
    tax_graph = model.TaxStruct(tax_pairs)
    acc, mrr, wu_p = score(results, tax_graph, trues)
    print("acc:", acc)
    print("mrr:", mrr)
    print("wu_p: ", wu_p)
