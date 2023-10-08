#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.ERROR)

from argparse import ArgumentParser
import grok
import subprocess

parser = ArgumentParser()
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--gpu', type=int, default=0)

trpcts = [50, 25, 10]
batchfracs = [0.5, 0.4, 0.3, 0.2]
operators = grok.data.VALID_OPERATORS.keys()

def format_command(name, gpu, batchfrac, trpct, operator):
    logdir = f'{name}/run-{operator}-batchfrac-{batchfrac}-trpct-{trpct}'
    return f'./train.py --logdir={logdir} --gpu={gpu} --batchsize={batchfrac} --train_data_pct={trpct} --math_operator={operator} weight_decay=1.0'

def plan(args):
    name = args.name
    gpu = args.gpu

    for trpct in trpcts:
        for batchfrac in batchfracs:
            for operator in operators:
                yield format_command(name, gpu, batchfrac, trpct, operator)

def main():
    args = parser.parse_args()
    print(f'Running experiment {args.name}')
    for cmd in plan(args):
        print(f'Running {cmd}')
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f'Error running {cmd}')
            print(result)
            break
        else:
            print(result)
            print(f'Success running {cmd}')

if __name__ == '__main__':
    main()
