import argparse

'''
usage: run.py [-h] [--a A]
An argparse for image classification package!
optional arguments:
  -h, --help  show this help message and exit
  -s --step   choices=["tfrecord", "load_data", "train", "fine_tune", "validate", "predict"]
  -a --assess boolean to check if the process needs to be assessed. for example, check the imported dataset or assess the accuracy of trained model
'''

parser = argparse.ArgumentParser(description='An argparse for image classification package!')
parser.add_argument("-s", "--step", required=True, type=str, help="which step to run? (tfrecord/load_data/train/fine_tune/train_and_fine_tune/predict)", choices=["tfrecord", "load_data", "train", "fine_tune", "train_and_fine_tune", "predict"])
parser.add_argument("-a", "--assess", required=False, type=bool, default=True, help="Do you need to asses the process(data/model/etc) that you ran")

args = parser.parse_args()

if args.step == "tfrecord":
    from  tfrecords import tfrecord
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
