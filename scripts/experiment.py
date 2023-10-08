#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.ERROR)

from argparse import ArgumentParser
import grok
import subprocess

parser = grok.training.add_args()
parser.add_argument('--name', type=str, default='experiment')

trpcts = [50, 25, 10]
batchfracs = [0.5, 0.4, 0.3, 0.2]
operators = grok.data.VALID_OPERATORS.keys()
operators_dict = grok.data.VALID_OPERATORS

def format_command(batchfrac, trpct, operator, args):
    logdir = f'{args.name}/run-{operators_dict[operator]}-batchfrac-{batchfrac}-trpct-{trpct}'
    return f'python scripts/train.py --logdir={logdir} --gpu={args.gpu} --batchsize={batchfrac} --train_data_pct={trpct} --math_operator={operator} --weight_decay={args.weight_decay} --max_steps={args.max_steps}'

def plan(args):
    for trpct in trpcts:
        for batchfrac in batchfracs:
            for operator in operators:
                yield format_command(batchfrac, trpct, operator, args)

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
