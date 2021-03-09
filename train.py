import argparse
import json
import model


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--config', type=str, help="path of config file", required=False, default=None)
    args, _ = arg_parser.parse_known_args()
    with open(args.config, mode='r', encoding='utf-8') as fp:
        train_args = json.load(fp)
    trainer = model.Trainer(args)
    trainer.train()
    trainer.save_model()
