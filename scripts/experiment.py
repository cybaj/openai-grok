#!/usr/bin/env python
from argparse import ArgumentParser
import logging
import os
import subprocess

import grok

logging.basicConfig(level=logging.ERROR)


parser = grok.training.add_args()
parser.add_argument("--name", type=str, default="experiment")

trpcts = [50, 25, 10]
# trpcts = [25, 10]
# batchfracs = [0.5, 0.4, 0.3, 0.2]
batchfracs = [0.2]
operators = grok.data.VALID_OPERATORS.keys()
operators_dict = grok.data.VALID_OPERATORS


def format_command(name, batchfrac, trpct, operator, args):
    logdir = (
        f"{name}/run-{operators_dict[operator]}-batchfrac-{batchfrac}-trpct-{trpct}"
    )
    cmd = f"python scripts/train.py --logdir={logdir} --batchsize={batchfrac} --train_data_pct={trpct} --math_operator={operator} --weight_decay=1.0 --max_steps {args.max_steps}"
    if args.duplication:
        cmd += f" --duplication={args.duplication}"
    if args.gpu:
        cmd += f" --gpu={args.gpu}"
    return cmd


def plan(args):
    name = args.name

    for trpct in trpcts:
        for batchfrac in batchfracs:
            for operator in operators:
                yield format_command(name, batchfrac, trpct, operator, args)


def save_hyperparameters(args):
    expt_dir = f"{args.name}"
    try:
        os.makedirs(expt_dir)
        hparams_file = os.path.join(expt_dir, "hparams.yaml")

        with open(hparams_file, "w") as fp:
            argument_dict = vars(args)
            for key, value in argument_dict.items():
                fp.write(f"{key}: {value}\n")
    except:
        raise Exception("couldn't make directory")


def main():
    args = parser.parse_args()
    print(f"Running experiment {args.name}")
    save_hyperparameters(args)
    for cmd in plan(args):
        print(f"Running {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Error running {cmd}")
            print(result)
            break
        else:
            print(result)
            print(f"Success running {cmd}")


if __name__ == "__main__":
    main()
