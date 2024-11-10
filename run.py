import os
import sys
from argparse import ArgumentParser

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../.."))
from basicts import launch_training


def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")

    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS08_itrvirlong01.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_itrvirlong01.py", help="training config")
    parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_METR_itr15_12.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_itrvirlong01.py", help="training config")

    # parser.add_argument("-c", "--cfg", default="stdmae/STNorm_PEMS04_test.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_test.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS03.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_test_ode05.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_test.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_itr02.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_itr12.py", help="training config")
    # parser.add_argument("-c", "--cfg", default="stdmae/STDMAE_PEMS04_test_odeInfo01.py", help="training config")
    parser.add_argument("--gpus", default="0", help="visible gpus")
    # parser.add_argument("--gpus", default="cpu", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    launch_training(args.cfg, args.gpus)

