"""this module contains tests for the dktt_light.model module"""

import pytest
import numpy as np
import tensorflow as tf

from dktt_light import model


@pytest.fixture
def time_blocks():
    all_time = tf.constant(
        [[0, 0, 1, 1, 1],
         [1, 1, 2, 2, 3]],
        dtype=tf.float32)

    time_block1 = all_time[:, :-1]
    time_block2 = all_time[:, 1:]

    return time_block1, time_block2


@pytest.fixture
def params():
    return {
        'hidden_size': 8,
        'confidence': 1,
        'temperature': 1,
        'time_unit': 1,
        'layer_postprocess_dropout': 0.1,
        'relu_dropout': 0.1,
        'attention_dropout': 0.1,
        'num_heads': 1,
        'shared_weights': False,
        'filter_size': 8,
        'kernal_size': 8,
        'mask_out': 'future',
        'num_encoder_blocks': 1,
        'num_decoder_blocks': 1,
        'q_matrix_trainable': True,
        'item_difficulty': False,
        'q_matrix': np.ones([8, 8], dtype=np.float32),
        }


@pytest.fixture
def inputs():
    batch_size = 16
    seq_len = 32
    item_vocab_size = 16
    time_max = 100

    items = np.random.randint(
        low=2, high=item_vocab_size, size=(batch_size, seq_len))
    times = np.random.uniform(0., time_max, size=(batch_size, seq_len))
    times = np.sort(times)

    return {
        'encoder_items': tf.constant(items[:, :-1], dtype=tf.int32),
        'encoder_times': tf.constant(times[:, :-1], dtype=tf.float32),
        'decoder_items': tf.constant(items[:, 1:], dtype=tf.int32),
        'decoder_times': tf.constant(times[:, 1:], dtype=tf.float32),
    }


def test_get_block_attention_bias_not_allow_current(time_blocks):
    time_block1, time_block2 = time_blocks

    if time_block1.dtype == tf.float16:
        neg_inf = model._NEG_INF_FP16
    elif time_block1.dtype == tf.float32:
        neg_inf = model._NEG_INF_FP32

    # shape = [2, 4, 4]
    expect = np.array(
        [
            [
                [neg_inf, neg_inf, neg_inf, neg_inf],
                [0., 0., neg_inf, neg_inf],
                [0., 0., neg_inf, neg_inf],
                [0., 0., neg_inf, neg_inf],
            ],
            [
                [neg_inf, neg_inf, neg_inf, neg_inf],
                [0., 0., neg_inf, neg_inf],
                [0., 0., neg_inf, neg_inf],
                [0., 0., 0., 0.],
            ],
    ])

    result = model.get_block_attention_bias(
        time_block1,
        time_block2,
        allow_current=False,
        dtype=time_block1.dtype).numpy()[:, 0, :, :]

    assert np.allclose(expect, result)


def test_get_block_attention_bias_allow_current(time_blocks):
    time_block1, time_block2 = time_blocks

    if time_block1.dtype == tf.float16:
        neg_inf = model._NEG_INF_FP16
    elif time_block1.dtype == tf.float32:
        neg_inf = model._NEG_INF_FP32

    # shape = [2, 4, 4]
    expect = np.array(
        [
            [
                [0., 0., neg_inf, neg_inf],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
            [
                [0., 0., neg_inf, neg_inf],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
    ])

    result = model.get_block_attention_bias(
        time_block1,
        time_block2,
        allow_current=True,
        dtype=time_block1.dtype).numpy()[:, 0, :, :]
    assert np.allclose(expect, result)


def test_get_time_block_attention_bias(params):
    params['mask_out'] = 'time_block'
    dktt_model = model.DKTTLight(params)

    # super simple inputs
    inputs = {
        'encoder_items': tf.constant([[2, 2], [3, 3]], dtype=tf.int32),
        'encoder_times': tf.constant([[0, 1], [1, 1]], dtype=tf.float32),
        'decoder_items': tf.constant([[2, 2], [3, 3]], dtype=tf.int32),
        'decoder_times': tf.constant([[1, 1], [1, 2]], dtype=tf.float32),
    }

    enc_self_attn_bias, enc_dec_attn_bias, dec_self_attn_bias = dktt_model.get_attention_bias(
        inputs)

    exp_enc_self_attn_bias = np.array(
        [
            [
                [
                    [0., model._NEG_INF_FP32],
                    [0., 0.]
                ]
            ],
            [
                [
                    [0., 0.],
                    [0., 0.],
                ]
            ]
        ],
        dtype=np.float32)

    exp_enc_dec_attn_bias = np.array(
        [
            [
                [
                    [0., model._NEG_INF_FP32],
                    [0., model._NEG_INF_FP32]
                ]
            ],
            [
                [
                    [model._NEG_INF_FP32, model._NEG_INF_FP32],
                    [0., 0.],
                ]
            ]
        ],
        dtype=np.float32)

    exp_dec_self_attn_bias = np.array(
        [
            [
                [
                    [0., 0.],
                    [0., 0.]
                ]
            ],
            [
                [
                    [0., model._NEG_INF_FP32],
                    [0., 0.],
                ]
            ]
        ],
        dtype=np.float32)

    assert np.allclose(exp_enc_self_attn_bias, enc_self_attn_bias)
    assert np.allclose(exp_enc_dec_attn_bias, enc_dec_attn_bias)
    assert np.allclose(exp_dec_self_attn_bias, dec_self_attn_bias)

