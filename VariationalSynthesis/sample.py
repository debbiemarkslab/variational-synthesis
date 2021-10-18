import argparse
import configparser
from datetime import datetime
import os
import pickle
import torch

from pyro.contrib.mue.dataloaders import write
from VariationalSynthesis import bio_utils as bu
from VariationalSynthesis.model import SynthesisModel


def main(config, args):
    # Setup.
    params_file = config['results']['params_file']
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_folder = os.path.join(args.out_folder, 'logs', time_stamp)
    os.mkdir(out_folder)

    # Load model.
    with open(params_file, 'rb') as pr:
        model = pickle.load(pr)

    # Draw samples.
    torch.manual_seed(args.rng_seed)
    drawn = 0
    batch_size = config['train']['batch_size']
    if batch_size == 'None':
        batch_size = args.nsamples
    else:
        batch_size = int(batch_size)
    sample_file = os.path.join(out_folder, 'samples.fasta')
    while drawn < args.nsamples:
        ntodraw = min([batch_size, args.nsamples - drawn])
        drawn += ntodraw
        # Draw.
        sample = model.get_samples(ntodraw)
        # Compute log probability.
        if args.compute_logp:
            batch_logps = model.get_log_probs(sample).cpu()
        else:
            batch_logps = None
        # Save samples.
        write(sample, bu.alphabets['aa'], sample_file,
              truncate_stop=args.truncate_stop, append=True,
              scores=batch_logps)

    # Save info.
    config['synth_samples'] = dict()
    config['synth_samples']['nsamples'] = str(args.nsamples)
    config['synth_samples']['truncate_stop'] = str(args.truncate_stop)
    config['synth_samples']['compute_logp'] = str(args.compute_logp)
    config['general']['synth_samples'] = str(sample_file)
    with open(os.path.join(out_folder, 'config.cfg'), 'w') as cw:
        config.write(cw)


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
                description="Sample from synthesis model.")
    parser.add_argument("configPath",
                        help=('Path to config file for ' +
                              'trained synthesis model.'))
    parser.add_argument("--nsamples", default=10, type=int,
                        help="Number of samples to draw.")
    parser.add_argument("--truncate-stop", action='store_true', default=False,
                        help='Truncate samples at stop symbol.')
    parser.add_argument("--compute-logp", action='store_true', default=False,
                        help='Compute and save log probability of samples.')
    parser.add_argument("--rng-seed", default=0, type=int)
    parser.add_argument("--out-folder", default='.')
    args = parser.parse_args()

    # Read config.
    config = configparser.ConfigParser()
    config.read(args.configPath)

    main(config, args)
