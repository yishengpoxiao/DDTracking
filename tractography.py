"""Train DDTracking or track fibers using an explicit YAML configuration."""

import argparse
import logging

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to a YAML configuration file')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true', help='Train a model')
    mode.add_argument('--track', action='store_true', help='Track using a trained model')
    args = parser.parse_args(argv)

    with open(args.config, encoding='utf-8') as stream:
        params = yaml.safe_load(stream)
    if not isinstance(params, dict):
        parser.error('The configuration must be a YAML mapping')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(args.config + '.log'), logging.StreamHandler()])
    logger = logging.getLogger('DDTracking')
    logger.setLevel(logging.DEBUG)
    if args.train:
        from trainer.trainer import Trainer

        trainer = Trainer(logger=logger, params=params)
        try:
            return trainer.train()
        finally:
            trainer.destroy_ddp()
    else:
        from tracker.tracker import Tracker

        return Tracker(logger=logger, params=params).track()


if __name__ == '__main__':
    main()
