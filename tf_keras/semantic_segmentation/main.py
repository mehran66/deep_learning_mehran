from __future__ import print_function
import datetime
print("__author__ = Mehran Ghandehari")
print(f"__date__ = {datetime.datetime.now()}")
print("__license__ = Feel free to copy :)")
import sys; print(sys.version)
import platform; print(platform.platform())
import tensorflow; print(f'tensorflow version: {tensorflow.__version__}')

import argparse

'''
usage: run.py [-h] [--a A]
An argparse for image segmentation package!
optional arguments:
  -h, --help  show this help message and exit
  -s --step   choices=["preprocess", "patchify", "tfrecord", "load_data", "train", "fine_tune", "train_and_fine_tune", "predict", "postprocess"]
  
  -a --assess boolean to check if the process needs to be assessed. for example, check the imported dataset or assess the accuracy of trained model
'''

parser = argparse.ArgumentParser(description='An argparse for image classification package!')
parser.add_argument("-s", "--step", required=True, type=str, help="which step to run? (preprocess/patchify/tfrecord/load_data/train/fine_tune/train_and_fine_tune/predict/postprocess)", choices=["preprocess", "patchify", "tfrecord", "load_data", "train", "fine_tune", "train_and_fine_tune", "predict", "postprocess"])
parser.add_argument("-a", "--assess", required=False, type=bool, default=True, help="Do you need to asses the process(data/model/etc) that you ran")

args = parser.parse_args()

if args.step == "preprocess":
    from preprocess import preprocess
    preprocess(assess_data=args.assess)

if args.step == "patchify":
    from patchify import patchify
    patchify(assess_data=args.assess)

if args.step == "tfrecord":
    from tfrecords import tfrecord
    tfrecord(assess_data=args.assess)

if args.step == "load_data":

    from load_data import load_data
    load_data(assess_data=args.assess)

if args.step == "train":

    from train import train
    train(assess_model = args.assess)

if args.step == "fine_tune":

    from fine_tune import fine_tune
    fine_tune(assess_model = args.assess)

if args.step == "train_and_fine_tune":

    from train import train
    train(assess_model=args.assess)

    from fine_tune import fine_tune
    fine_tune(assess_model = args.assess)

if args.step == "predict":

    from predict import predict
    predict()

if args.step == "postprocess":

    from postprocess import postprocess
    postprocess()

