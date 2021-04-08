import logging
from argparse import ArgumentParser

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

import preprocessing
import dktt_light_config
import model as dktt_light_model
from losses import PaddedBinaryCrossentropyLoss

logger = logging.getLogger(__file__)
# todo: set level after parase user config
logger.setLevel(logging.INFO if dktt_light_config.VERBOSE == 1 else logging.WARNING)
console = logging.StreamHandler()
console.setLevel(logging.INFO if dktt_light_config.VERBOSE == 1 else logging.WARNING)
logger.addHandler(console)



def evaluate(test_inputs, test_y, model, test_size, batch_size=128):
    pt = 0
    test_probs = []

    while pt < test_size:
        batch_inputs = dict([(key, val[pt: pt + batch_size]) for key, val in test_inputs.items()])
        batch_probs = model(batch_inputs, training=False)
        batch_probs = batch_probs.numpy()

        test_probs.append(batch_probs)
        pt += batch_size

    test_probs = np.concatenate(test_probs, axis=0)

    # cacluate metrics
    y_true = test_y.reshape(-1)
    y_pred = test_probs.reshape(-1)

    mask = y_true != 0
    y_true_masked = y_true[mask] - 1  # so 2 -> 1, 1 -> 0
    y_pred_masked = y_pred[mask]

    return roc_auc_score(y_true_masked, y_pred_masked)


class ModelCompiler:
    def __init__(
            self,
            initial_learning_rate,
            decay_steps,
            decay_rate,
            smoothing,
    ):
        self.optimizer_params = {
            'initial_learning_rate': initial_learning_rate,
            'decay_steps': decay_steps,
            'decay_rate': decay_rate,
            'staircase': True}

        self.loss_params = {
            'smoothing': smoothing,
        }

    def get_compiler_params(self):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(**self.optimizer_params)

        compile_params = {
            'optimizer': tf.keras.optimizers.Adam(learning_rate=lr),
            'loss': PaddedBinaryCrossentropyLoss(**self.loss_params)
        }

        return compile_params


def train_one_cv(dataset, data_dir, cv_idx, data_config, model_params, fit_params, model_compiler):
    logger.info(f'\n cv {cv_idx} starts for {dataset}\n')
    data = preprocessing.DataSet(
        dataset,
        data_dir,
        max_len=dktt_light_config.MAX_LEN,
        cv_idx=cv_idx,
        **data_config)

    data.load_data()

    if cv_idx == 0:
        data.describe_data()  # only need to describe the data for the first time training

    (train_inputs, train_targets), (test_inputs, test_targets), item_skill_mapping = data.preprocess()

    # overwrite q_matrix before training
    model_params['params']['q_matrix'] = item_skill_mapping

    model = dktt_light_model.DKTTLight(**model_params)
    logger.info('model is created')

    compiler_params = model_compiler.get_compiler_params()
    model.compile(**compiler_params)
    logger.info('model is compiled')

    logger.info('start fitting')
    model.fit(
        train_inputs,
        train_targets,
        verbose=dktt_light_config.VERBOSE,
        **fit_params)

    score = evaluate(
        test_inputs,
        test_targets,
        model,
        test_targets.shape[0],
        batch_size=128)
    # todo: change this level
    logger.info(f'test auc = {score} in cv = {cv_idx}')
    logger.info(f'\n cv {cv_idx} finished for {dataset}\n')

    if cv_idx == 4:
        logger.info(f'{model.summary()}')

    return score


def train_cv(dataset, opt):
    data_config = dktt_light_config.config[dataset]

    if dataset == 'assist2017':
        model_params, fit_params, compile_params = dktt_light_config.assist_2017_params
    elif dataset == 'stat2011':
        model_params, fit_params, compile_params = dktt_light_config.stat_2011_params
    else:
        raise NotImplementedError

    overwrite_default_args(model_params['params'], opt)
    overwrite_default_args(fit_params, opt)
    overwrite_default_args(compile_params, opt)

    model_compiler = ModelCompiler(**compile_params)

    logger.info(f'model params are {model_params}')
    logger.info(f'fit params are {fit_params}')
    logger.info(f'optimizer params = {model_compiler.optimizer_params}')
    logger.info(f'loss params = {model_compiler.loss_params}')

    cv_scores = []
    for cv_idx in range(dktt_light_config.CV_NUM):
        cv_score = train_one_cv(
            dataset,
            (opt.dir or dktt_light_config.DATA_DIR),
            cv_idx,
            data_config,
            model_params,
            fit_params,
            model_compiler)
        cv_scores.append(cv_score)

    # todo: change this level
    logger.info(f'average test auc for {dataset} is {sum(cv_scores) / dktt_light_config.CV_NUM}')
    return cv_scores


def get_parser():
    parser = ArgumentParser()

    # not all arguments are exposed to change for users
    # all arguments controlling the behavior of the model is in dktt_light_config,
    # for complete control over the model, modify dktt_light_config
    parser.add_argument(
        '--dataset',
        choices=['assist2017', 'stat2011'],
        required=True,
        help='dataset to run the model'
    )

    parser.add_argument(
        '--dir',
        help='folder that hold data'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        choices=[0, 1],
        type=int,
        help='model verbosity, set to 1 for more information; 0 to suppress information')

    # model args
    parser.add_argument(
        '--hidden_size',
        help='hidden size for transformer blocks',
        type=int,
    )

    parser.add_argument(
        '--confidence',
        help='confidence for the q_matrix',
        type=float,
    )

    parser.add_argument(
        '--temperature',
        help='temperature for the q_matrix ',
        type=float,
    )

    parser.add_argument(
        '--layer_postprocess_dropout',
        help='dropout for post process layer',
        type=float,
    )

    parser.add_argument(
        '--relu_dropout',
        help='dropout for the feedfoward network',
        type=float,
    )

    parser.add_argument(
        '--attention_dropout',
        help='dropout for the attention layer',
        type=float,
    )

    parser.add_argument(
        '--num_heads',
        help='number of heads in the attention layer',
        type=int,
    )

    parser.add_argument(
        '--filter_size',
        help='hidden layer size for the feedforward layer',
        type=int,
    )

    parser.add_argument(
        '--kernal_size',
        help='hidden size for the vector representing time distance between interactions',
        type=int,
    )

    parser.add_argument(
        '--num_encoder_blocks',
        help='number of encoder blocks',
        type=int,
    )

    parser.add_argument(
        '--num_decoder_blocks',
        help='number of decoder blocks',
        type=int,
    )

    # fit args
    parser.add_argument(
        '--batch_size',
        type=int,
        help='training batch size'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        help='number of epochs to train'
    )

    parser.add_argument(
        '--validation_split',
        type=float,
        help='poriton of training with hold for validation'
    )

    parser.add_argument(
        '--initial_learning_rate',
        type=float,
        help='initial learning rate'
    )

    parser.add_argument(
        '--decay_steps',
        type=int,
        help='decay steps for learning rate'
    )

    parser.add_argument(
        '--decay_rate',
        type=float,
        help='decay rate at each decay'
    )

    return parser


def overwrite_default_args(default_config, opt):
    for key in default_config:
        user_opt = getattr(opt, key, None)
        if user_opt is not None:
            default_config[key] = user_opt


if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    train_cv(opt.dataset, opt)




