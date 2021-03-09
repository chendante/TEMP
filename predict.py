import argparse


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--taxo_path', type=str, help="path of existing taxonomy dataset", required=False,
                            default="./data/raw_data/science_wordnet_en.taxo")
