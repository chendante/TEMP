import argparse
import model


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--taxo_path', type=str, help="path of existing taxonomy dataset", required=False,
                            default="./data/train/science_train.taxo")
    arg_parser.add_argument('--model_path', type=str, help="path of trained model", required=False,
                            default="./data/model/")
    arg_parser.add_argument('--terms', type=str, help="path of prediction terms", required=False,
                            default="./data/eval/science_eval.terms")
    arg_parser.add_argument('--dict_path', type=str, help="path of dictionary file", required=False,
                            default="./data/")
    arg_parser.add_argument('--output', type=str, help="path of result file", required=False,
                            default="./data/result/science.results")
    arg_parser.add_argument('--padding_max', type=int, help="max num of tokens", required=False,
                            default=256)
    args, _ = arg_parser.parse_known_args()
    e = model.Eval(args)
    results = e.predict()
    e.save_results(results)
