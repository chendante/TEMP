import model
import util
import torch
import scorer


def run(seed=0):
    train_args = util.DotDict(dict(
        # paths
        taxo_path="./data/train/science_train.taxo",
        pretrained_path="./data/models/bert-base-uncased/",
        dic_path="./data/dic/dic.json",
        save_path="./data/models/trained_science_temp/",
        log_path="./data/log/",
        # config
        seed=seed,
        margin_beta=0.25,
        epochs=55,
        batch_size=32,
        lr=3e-5,
        eps=1e-8,
        padding_max=120,
        log_label="science",
    ))
    # torch.cuda.set_device(1)
    trainer = model.Trainer(train_args)
    trainer.train()
    trainer.save_model()
    eval_args = util.DotDict(dict(
        # paths
        taxo_path="./data/train/science_train.taxo",
        model_path="./data/models/trained_science_temp/",
        dic_path="./data/dic/dic.json",
        terms="./data/eval/science_eval.terms",
        output="./data/result/science.results",
        # config
        padding_max=120,
    ))
    e = model.Eval(eval_args)
    results = e.predict()
    with open("./data/eval/science_eval.taxo", encoding='utf-8') as fp:
        lines = fp.readlines()
    trues = [line.strip() for line in lines]
    acc, mrr, wu_p = scorer.score(results, trainer.tax_graph, trues)
    return acc, mrr, wu_p


if __name__ == '__main__':
    run()
