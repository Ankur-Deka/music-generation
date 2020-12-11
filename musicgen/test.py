import argparse
import torch
from torch import nn
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer  import to_device


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    dl_args = config['data_loader']['args']
    dl_args['shuffle'] = False
    dl_args['batch_size'] = 512
    dl_args['validation_split'] = 0
    dl_args['training'] = False
    dl_args['num_workers'] = 2

    data_loader = getattr(module_data, config['data_loader']['type'])(
            **dl_args
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1 and False:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            data = to_device(data, device)
            if isinstance(model, nn.DataParallel):
                notes = model.module.generate_notes(data)
            else:
                notes = model.generate_notes(data)

            # save sample images, or do something with output here
            #print(len(notes), notes[0].shape)
            print(notes.shape)

        data_loader.dataset.convert_to_midi(notes)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--run_id', type=str, default='test')

    config = ConfigParser.from_args(args)
    main(config)
