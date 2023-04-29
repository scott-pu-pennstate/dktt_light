import os


CV_NUM = 5
MAX_LEN = 128
HOME_DIR = os.path.dirname(
    os.path.dirname(__file__))
DATA_DIR = os.path.join(
    HOME_DIR,
    'data')
NAME_CONVENTION = '{}-cv-{}-{}.csv'
VERBOSE = 1

config = {
    'assist2017': {
        'id_col': 'ITEST_id',
        'time_col': 'startTime',
        'prob_col': 'problemId',
        'skill_col': 'skill',
        'score_col': 'correct',
        'skill_sep': '~',
    },
    'stat2011': {
        'id_col': 'Anon Student Id',
        'time_col': 'Time',
        'prob_col': 'problem',
        'skill_col': 'duplicated_problem',
        'score_col': 'Outcome',
        'skill_sep': '~',
    },
}

assist_2017_params = [
    # model params
    {'params': {
        'hidden_size': 32,
        'confidence': 2,
        'temperature': 0.5,
        'time_unit': 1,
        'layer_postprocess_dropout': 0.1,
        'relu_dropout': 0.1,
        'attention_dropout': 0.1,
        'num_heads': 4,
        'shared_weights': False,
        'filter_size': 64,
        'kernal_size': 8,
        'mask_out': 'future',
        'num_encoder_blocks': 4,
        'num_decoder_blocks': 0,
        'q_matrix_trainable': True,
        'item_difficulty': False
        }},

    # fit params
    {'batch_size': 128,
     'epochs': 200,
     'validation_split': 0.1},

    {
        'initial_learning_rate': 0.001,
        'decay_steps': 100,
        'decay_rate': 1.,
        'smoothing': 0.1,
    }
]

stat_2011_params = [
    {'params': {
        'hidden_size': 32,
        'confidence': 4,
        'temperature': 0.5,
        'time_unit': 1,
        'layer_postprocess_dropout': 0.15,
        'relu_dropout': 0.15,
        'attention_dropout': 0.15,
        'num_heads': 8,
        'shared_weights': False,
        'filter_size': 64,
        'kernal_size': 16,
        'mask_out': 'future',
        'num_encoder_blocks': 2,
        'num_decoder_blocks': 0,
        'q_matrix_trainable': False,
        'item_difficulty': False
        }
    },

    {'batch_size': 128,
     'epochs': 50,
     'validation_split': 0.1},

    {
        'initial_learning_rate': 0.001,
        'decay_steps': 100,
        'decay_rate': 1.,
        'smoothing': 0.1,
    }
]
