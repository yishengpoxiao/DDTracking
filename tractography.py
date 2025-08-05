import yaml

from trainer.trainer import Trainer
from tracker.tracker import Tracker
import argparse
import os
import logging

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=False, help='Path to .json configuration file')
    parser.add_argument('--train', action="store_true", required=False, help='Whether to start training phase')
    parser.add_argument('--track', action="store_true", required=False, help='Whether to start inference phase')
    args = parser.parse_args()

    # Logging control
    if args.config is not None:
        log_path = args.config + '.log'
    else:
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        log_path = os.path.join(dname, '.log')

    logging.basicConfig(filename=log_path, filemode='a', format='%(asctime)s %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

    # Get parameter file from arguments or from default source
    with open(args.config, 'r') as f:
        params = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Train model
    if args.train:
        trainer = Trainer(logger=logger, params=params)
        train_performance = trainer.train()

    # Run tractography using a trained model
    if args.track:
        tracker = Tracker(logger=logger, params=params)
        tractogram = tracker.track()
