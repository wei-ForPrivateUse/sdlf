import os
import sys
import time
import torch
import torchplus

from tqdm import tqdm
from tensorboardX import SummaryWriter
from sdlf.optimizer import optimizer_builder, lr_scheduler_builder
from sdlf.models.net import Net
from sdlf.ops.common import get_class, read_config, Logger, flatten_deep_dict, try_restore_latest_checkpoints_

MAJOR_VERSION = 1
MINOR_VERSION = 2
PATCH_VERSION = 1


def _evaluate_helper(dataloader,
                     net,
                     eval_fn,
                     eval_ext_args):
    # check evaluation function
    if not eval_fn:
        print('evaluation function not specified, skipped. ', flush=True)
        return

    # save and set training status
    net_training_bk = net.training
    net.train(False)

    # get labels and predictions
    print('*****************************************', flush=True)
    print('generating predicted outputs... ', flush=True)
    sys.stdout.flush()
    label_list, pred_list = [], []
    for example_val in tqdm(dataloader):
        with torch.no_grad():
            label_out, pred_out, _ = net(example_val)
        label_list.append(label_out)
        pred_list.append(pred_out)
    sys.stderr.flush()
    print('done.', flush=True)
    print('evaluating... ', flush=True)
    eval_res = eval_fn(label_list, pred_list, eval_ext_args)
    print('done.', flush=True)
    print(eval_res, flush=True)
    print('*****************************************', flush=True)

    # resume training status
    net.train(net_training_bk)


def evaluate(dataset_cfg_path,
             model_cfg_path,
             train_cfg_path,
             model_path,
             dataset_section='val'):
    # get configurations
    dataset_cfg = read_config(dataset_cfg_path)
    model_cfg = read_config(model_cfg_path)
    train_cfg = read_config(train_cfg_path)

    # prepare dataset
    dataset = get_class(dataset_cfg[dataset_section]['class'])(dataset_cfg[dataset_section])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_cfg[dataset_section]['batch_size'],
        shuffle=False,
        num_workers=dataset_cfg[dataset_section]['num_workers'],
        pin_memory=False,
        collate_fn=get_class(dataset_cfg[dataset_section]['collate_fn']),
    )

    # prepare network model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(model_cfg['models']).to(device)
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)

    # set other parameters
    eval_fn = None if not train_cfg['training']['eval_fn'] else get_class(train_cfg['training']['eval_fn'])
    eval_ext_args = None if not eval_fn else train_cfg['training']['eval_ext_args']

    # evaluation
    _evaluate_helper(dataloader, net, eval_fn, eval_ext_args)


def train(dataset_cfg_path,
          model_cfg_path,
          train_cfg_path,
          result_dir,
          project_name='',
          display_step=50,
          resume=False):
    """
    main entrance for training

    :param dataset_cfg_path: configuration file for dataset
    :param model_cfg_path: configuration file for model
    :param train_cfg_path: configuration file for training
    :param result_dir: directory for saving models and logs
    :param project_name: name of project to be displayed
    :param display_step: display logs every display steps
    :param resume: try resuming training from checkpoints, if specified
    :return: None
    """
    # resume check & model dir
    if os.path.exists(result_dir) and not resume:
        raise Exception('result_dir exists, but resume=False')
    if not os.path.exists(result_dir) and resume:
        raise Exception('result_dir dose not exist, but resume=True')
    if not os.path.exists(result_dir) and not resume:
        os.makedirs(result_dir)

    # initialize logger
    logger = Logger(os.path.join(result_dir, 'log.txt'))
    logger.bind()

    # log info
    if not project_name:
        project_name = 'Unspecified'
    print(f'{project_name}, powered by Simple Deep Learning Framework v{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}', flush=True)

    # get configurations
    dataset_cfg = read_config(dataset_cfg_path)
    model_cfg = read_config(model_cfg_path)
    train_cfg = read_config(train_cfg_path)

    dataset_train_config = dataset_cfg['train']
    dataset_val_config = dataset_cfg['val']
    model_config = model_cfg['models']
    optimizer_config = train_cfg['optimizer']
    training_config = train_cfg['training']

    # log info
    print(dataset_cfg, flush=True)
    print(model_cfg, flush=True)
    print(train_cfg, flush=True)

    # dataset
    dataset_train = get_class(dataset_train_config['class'])(dataset_train_config)
    dataset_val = get_class(dataset_val_config['class'])(dataset_val_config)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=dataset_train_config['batch_size'],
        shuffle=True,
        num_workers=dataset_train_config['num_workers'],
        pin_memory=False,
        collate_fn=get_class(dataset_train_config['collate_fn']),
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=dataset_val_config['batch_size'],
        shuffle=False,
        num_workers=dataset_val_config['num_workers'],
        pin_memory=False,
        collate_fn=get_class(dataset_val_config['collate_fn']),
    )

    # prepare network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(model_config).to(device)

    # build optimizers and lr_schedulers
    total_step = training_config['total_step']
    optimizers, lr_schedulers = [], []
    for idx, mod_lv_cfg in enumerate(optimizer_config):
        modules = mod_lv_cfg['modules']
        optimizers.append(optimizer_builder.build(mod_lv_cfg, net, modules, '{}_{}'.format(mod_lv_cfg['type'], idx)))
        lr_schedulers.append(lr_scheduler_builder.build(mod_lv_cfg['lr_scheduler'], total_step, optimizers[-1]))

    # try restore checkpoints
    resume_step = try_restore_latest_checkpoints_(result_dir, [net] + optimizers)

    # get training configurations
    save_step_list = training_config['save_step_list']
    eval_step_list = training_config['eval_step_list']
    eval_fn = None if not training_config['eval_fn'] else get_class(training_config['eval_fn'])
    eval_ext_args = None if not eval_fn else training_config['eval_ext_args']

    # initialization
    for optimizer in optimizers:
        optimizer.zero_grad()
    current_step = 0 if not resume_step else resume_step[0] + 1
    writer = SummaryWriter(logdir=os.path.join(result_dir, 'tensorboardX'))

    # main loop, start training
    while current_step < total_step:
        for example_train in train_loader:
            # start step timer
            t = time.time()
            torch.cuda.synchronize()

            # lr scheduler step
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step(current_step)

            # network forward
            loss_dict, loss_info, _ = net(example_train)

            # get loss & backward
            loss = loss_dict['loss']
            loss.backward()

            # clip grad & optimizer step
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            # stop step timer
            torch.cuda.synchronize()
            step_time = time.time() - t

            # display and write to tensorboardX
            if current_step % display_step == 0 and current_step > 0:
                print(f"@@@ step: {current_step} @@@ --- loss: {loss}, step_time: {step_time}", flush=True)
                print(loss_info, flush=True)
                flat_dict = flatten_deep_dict(loss_info)
                for key, value in flat_dict.items():
                    writer.add_scalars(key, value, current_step)

            # save checkpoints
            save_flag = (isinstance(save_step_list, list) and current_step in save_step_list) or \
                        (isinstance(save_step_list, int) and current_step % save_step_list == 0 and current_step > 0)
            if save_flag:
                torchplus.train.save_models(result_dir, [net] + optimizers, current_step)

            # evaluation
            eval_flag = (isinstance(eval_step_list, list) and current_step in eval_step_list) or \
                        (isinstance(eval_step_list, int) and current_step % eval_step_list == 0 and current_step > 0)
            if eval_flag:
                _evaluate_helper(val_loader, net, eval_fn, eval_ext_args)

            current_step += 1
            if current_step >= total_step:
                torchplus.train.save_models(result_dir, [net] + optimizers, current_step)
                _evaluate_helper(val_loader, net, eval_fn, eval_ext_args)
                break
    writer.close()

    # release logger
    logger.release()
