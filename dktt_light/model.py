# some of the layers reuse the code from tensorflow/models/nlp/transformer
from official.nlp.modeling import layers
from official.nlp.transformer import ffn_layer
from official.nlp.transformer.model_utils import get_padding_bias, get_decoder_self_attention_bias, _NEG_INF_FP32, _NEG_INF_FP16

import numpy as np
import tensorflow as tf
import math


def get_block_attention_bias(
        block_seq1,
        block_seq2,
        allow_current=False,
        dtype=tf.float32):
    r"""
    Parameters
    ----------
    block_seq1 : tf.Tensor
        a tensor of time block, shape = [batch_size, seq_len]
    block_seq2 : tf.Tensor
        a tensor of time block, shape = [batch_size, seq_len]
    allow_current : bool
        if True, allow seq to attention to current block
    dtype : tf.dtypes.DType
        default to tf.float32, also support tf.float16

    Returns
    -------
    attention_bias: tf.Tensor
        a tensor of shape [batch_size, 1, seq_len, seq_len], the semantic
        meaning of each dimension is [batch_size, num_heads, seq_len, seq_len]
    """

    neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32

    seq1 = tf.expand_dims(block_seq1, axis=1)
    seq2 = tf.expand_dims(block_seq2, axis=1)
    seq2_t = tf.transpose(seq2, perm=[0, 2, 1])

    # block smaller than current block is valid
    if allow_current:
        valid_locs = tf.cast(tf.greater_equal(seq2_t - seq1, 0), dtype=dtype)
    else:
        valid_locs = tf.cast(tf.greater(seq2_t - seq1, 0), dtype=dtype)

    valid_locs = tf.expand_dims(valid_locs, axis=1)
    decoder_bias = neg_inf * (1.0 - valid_locs)

    return decoder_bias


class LayerNormalization(tf.keras.layers.Layer):
    """ apply layer normalization"""
    def __init__(self, hidden_size, epsilon=1e-6, dtype='float32', trainable=False):
        super(LayerNormalization, self).__init__(dtype=dtype)
        self.hidden_size = hidden_size
        if trainable:
            # by default, tf.keras.layers.LayerNormalization has parameters trained from data
            self.layer = tf.keras.layers.LayerNormalization(
                epsilon=epsilon, dtype=dtype)
        else:
            self.layer = self._normalize_inputs
            self.epsilon = epsilon

    def get_config(self):
        parent_config = super(LayerNormalization, self).get_config()
        self_config = {'hidden_size': self.hidden_size}
        return dict({**parent_config, **self_config})

    def _normalize_inputs(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + self.epsilon)

    def call(self, x):
        return self.layer(x)


class PrePostProcessingWrapper(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params["layer_postprocess_dropout"]

    def build(self, input_shape):
        # Create normalization layer
        self.layer_norm = LayerNormalization(
            epsilon=1e-6, dtype="float32", hidden_size=self.params['hidden_size'])
        super(PrePostProcessingWrapper, self).build(input_shape)

    def get_config(self):
        return {
            "params": self.params,
        }

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]

        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_size: int,
            q_matrix: np.ndarray,  # map item to skill,
            q_matrix_trainable: bool,
            confidence: float,
            temperature: float,
            **kwargs) -> None:
        super(EmbeddingSharedWeights, self).__init__(**kwargs)

        # get vocab size
        item_vocab_size, skill_vocab_size = q_matrix.shape

        # skill embedding
        self.skill_shared_weights = self.add_weight(
            'skill_shared_weights',
            shape=[skill_vocab_size, hidden_size],
            trainable=True,
            initializer=tf.random_normal_initializer(mean=0., stddev=hidden_size ** -0.5))

        # initialize item-skill-mapping with  q-matrix
        low_confidence = 1 - confidence  # confidence in (0., 1.)
        init_q_matrix = confidence * q_matrix + (1. - q_matrix) * low_confidence
        init_q_matrix /= temperature  # temperature in (0., 1.)

        self.item_skill_mapping = self.add_weight(
            shape=[item_vocab_size, skill_vocab_size],
            trainable=q_matrix_trainable,
            initializer=tf.keras.initializers.Constant(init_q_matrix))

        self.params = {
            'q_matrix': q_matrix,
            'q_matrix_trainable': q_matrix_trainable,
            'confidence': confidence,
            'temperature': temperature,
            'hidden_size': hidden_size,
        }

    def build(self, input_shape):
        super(EmbeddingSharedWeights, self).build(input_shape)

    def get_config(self):
        return self.params

    def call(self, nested_inputs, mode='embedding'):
        if mode == 'embedding':
            return self._embedding(nested_inputs['items'])
        elif mode == 'linear':
            return self._linear(
                nested_inputs['states'],
                nested_inputs['query_items'])
        else:
            raise ValueError('mode {} is not valid'.format(mode))

    def _embedding(self, items, normalize=True):
        # item = (q, c), shape = [batch_size, seq_len]
        # get item-skill-mapping
        mapping = self.item_skill_mapping
        item_skill_mapping = tf.gather(mapping, items)  # shape = [batch_size, seq_len, skill_vocab]
        item_skill_mapping = tf.nn.softmax(item_skill_mapping, axis=-1)

        # shape = [batch_size, seq_len, hidden_size]
        item_embeddings = tf.einsum(
            'BLS, SH -> BLH',
            item_skill_mapping,
            self.skill_shared_weights,
        )
        # item_embeddings = tf.matmul(item_skill_mapping, self.skill_shared_weights)  # shape = [batch_size, seq_len, hidden_size]
        mask = tf.cast(tf.not_equal(items, 0), item_embeddings.dtype)
        item_embeddings *= tf.expand_dims(mask, -1)
        if normalize:
            item_embeddings *= self.params['hidden_size'] ** 0.5
            item_embeddings /= tf.reduce_sum(item_skill_mapping ** 2, axis=-1, keepdims=True) ** .5

        return item_embeddings

    def _linear(self, states, query_items):
        # query_items = (q, 2), 0 is for padding

        # get query embeddings, shape = [batch_size, seq_len, hidden_size]
        embeddings1 = self._embedding(query_items, normalize=False)      # embedding for (q, 2)
        query_equal_pad = tf.cast(tf.equal(query_items, 0), dtype=query_items.dtype)
        query_items2 = (query_items - 1) * (1 - query_equal_pad)
        embeddings2 = self._embedding(query_items2, normalize=False)  # embedding for (q, 2), use the fact that Encoder((q, 2)) = Encoder((q, 1)) + 1
        query_embeddings = tf.stack([embeddings1, embeddings2], axis=-2)

        # use dot product to calculate alignment between states and query items
        logits = tf.einsum(
            'BLNH, BLH -> BLN',
            query_embeddings,
            states)

        return logits[:, :, 0] - logits[:, :, 1]


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention with bias kernal"""
    def __init__(self, hidden_size, num_heads, attention_dropout, kernal_size, **kwargs):
        """ initialize Attention
        Argument:
            hidden_size (int), output dim
            num_heads (int), number of heads
            attention_dropout (float)
        """

        if hidden_size % num_heads:
            raise ValueError('hidden size must be divisible by num of heads')

        super(Attention, self).__init__(**kwargs)
        self.params = {
            'hidden_size': hidden_size,
            'num_heads': num_heads,
            'attention_dropout': attention_dropout,
            'kernal_size': kernal_size,
        }

    def build(self, input_shape):
        size_per_head = self.params['hidden_size'] // self.params['num_heads']

        self.query_dense_layer = layers.DenseEinsum(
            output_shape=(self.params['num_heads'], size_per_head),
            kernel_initializer='glorot_uniform',
            use_bias=False,
            name='query')

        self.key_dense_layer = layers.DenseEinsum(
            output_shape=(self.params['num_heads'], size_per_head),
            kernel_initializer='glorot_uniform',
            use_bias=False,
            name='key')

        self.value_dense_layer = layers.DenseEinsum(
            output_shape=(self.params['num_heads'], size_per_head),
            kernel_initializer='glorot_uniform',
            use_bias=False,
            name='value')

        self.output_dense_layer = layers.DenseEinsum(
            output_shape=self.params['hidden_size'],
            num_summed_dimensions=2,
            kernel_initializer='glorot_uniform',
            use_bias=False,
            name='output_transform')

        self.time_bias_hidden_layer = layers.DenseEinsum(
                output_shape=self.params['kernal_size'],
                kernel_initializer='glorot_uniform',
                use_bias=True,
                activation='relu', # # the choice of tanh makes sure the hidden layer is robust to large input time
        )

        self.time_bias_output_layer = layers.DenseEinsum(
                output_shape=1,
                kernel_initializer='glorot_uniform',
                use_bias=True,
                activation='linear')

        # todo: 1. should allow each head to have different retention rate
        # todo: 2. should pass in dtype
        super(Attention, self).build(input_shape)

    def get_config(self):
        return self.params

    def call(
            self,
            query_inputs,
            source_inputs,
            query_source_dist,
            bias,
            training):

        query = self.query_dense_layer(query_inputs)
        key = self.key_dense_layer(source_inputs)
        value = self.value_dense_layer(source_inputs)

        depth = (self.params['hidden_size'] // self.params['num_heads'])
        query *= depth ** -0.5

        # this is only need if not using sinusoide encoding
        query_source_dist = tf.expand_dims(query_source_dist, axis=-1)  # shape = [batch_size, q_seq_len, key_seq_len, 1]

        # query bias
        query_source_bias = self.time_bias_hidden_layer(query_source_dist)
        if training:
            query_source_bias = tf.nn.dropout(query_source_bias, rate=self.params['attention_dropout'])

        query_source_bias = self.time_bias_output_layer(query_source_bias)[:, :, :, 0]  # [batch_size, q_seq_len, key_seq_len]

        # key query agreement
        logits = tf.einsum("BTNH,BFNH->BNFT", key, query)
        query_source_bias = tf.expand_dims(query_source_bias, axis=1)
        logits += bias + query_source_bias  # mask out future and give more weight to recent item responses

        weights = tf.nn.softmax(logits, name="attention_weights")
        if training:
            weights = tf.nn.dropout(weights, rate=self.params['attention_dropout'])
        attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done --> [batch_size, length, hidden_size]
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class DecoderStack(tf.keras.layers.Layer):
    r"""this implement the encoder stack
    """

    def __init__(self, params, name='decoder_stack'):
        super(DecoderStack, self).__init__(name=name)
        self.params = params
        self.blocks = []

    def build(self, input_shape):
        r"""build the decoder stack."""
        self.output_normalization = LayerNormalization(
            hidden_size=self.params['hidden_size'])

        for i in range(self.params['num_decoder_blocks']):
            block = {}

            if self.params['shared_weights'] is True and (i > 0):
                block = self.blocks[i - 1]
            else:
                block['dec_self_attention'] = Attention(
                    hidden_size=self.params['hidden_size'],
                    num_heads=self.params['num_heads'],
                    attention_dropout=self.params['attention_dropout'],
                    kernal_size=self.params['kernal_size'],
                    name='dec_self_attn')

                block['enc_dec_attention'] = Attention(
                        hidden_size=self.params['hidden_size'],
                        num_heads=self.params['num_heads'],
                        attention_dropout=self.params['attention_dropout'],
                        kernal_size=self.params['kernal_size'],
                        name='enc_dec_attn')

                block['dec_feed_forward'] = ffn_layer.FeedForwardNetwork(
                        hidden_size=self.params['hidden_size'],
                        filter_size=self.params['filter_size'],
                        relu_dropout=self.params['relu_dropout'])

                # prepostproces
                block = dict([
                    (name, PrePostProcessingWrapper(layer, self.params)) for name, layer in block.items()])
            self.blocks.append(block)

        super(DecoderStack, self).build(input_shape)

    def call(self,
             decoder_inputs,
             encoder_outputs,
             decoder_times_dist,
             decoder_encoder_times_dist,
             dec_self_attn_bias,
             enc_dec_attn_bias,
             training,
             cache=None,
             decode_loop_step=None):
        r"""return the output of decoder layer stacks"""

        for n, block in enumerate(self.blocks):
            # run inputs through the sublayers.
            block_name = 'block_%d' % n

            with tf.name_scope(block_name):
                decoder_inputs = block['enc_dec_attention'](
                    decoder_inputs,
                    encoder_outputs,
                    query_source_dist=decoder_encoder_times_dist,
                    bias=enc_dec_attn_bias,
                    training=training)

                decoder_inputs = block['dec_feed_forward'](
                    decoder_inputs, training=training)

        if self.params['num_decoder_blocks'] == 0:
            return encoder_outputs
        return self.output_normalization(decoder_inputs)

    def get_config(self):
        return {
            'params': self.params,
        }


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, params, name='encoder_stack'):
        super(EncoderStack, self).__init__(name=name)
        self.params = params
        self.blocks = []

    def build(self, input_shape):
        self.output_normalization = LayerNormalization(
            hidden_size=self.params['hidden_size'])

        for i in range(self.params['num_encoder_blocks']):
            block = {}
            if self.params['shared_weights'] is True and (i > 0):
                block = self.blocks[i - 1]
            else:
                block['enc_self_attention'] = Attention(
                    hidden_size=self.params['hidden_size'],
                    num_heads=self.params['num_heads'],
                    attention_dropout=self.params['attention_dropout'],
                    kernal_size=self.params['kernal_size'],
                    name='enc_self_attn')

                block['enc_feed_forward'] = ffn_layer.FeedForwardNetwork(
                    hidden_size=self.params['hidden_size'],
                    filter_size=self.params['filter_size'],
                    relu_dropout=self.params['relu_dropout'])
                # prepostproces
                block = dict([
                  (name, PrePostProcessingWrapper(layer, self.params)) for name, layer in block.items()])

            self.blocks.append(block)

        super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params,
        }

    def call(
            self,
            encoder_inputs,
            encoder_time_dist,
            attention_bias,
            training):

        for n, block in enumerate(self.blocks):
            # run inputs through the sublayers.
            block_name = 'block_%d' % n

            with tf.name_scope(block_name):
                encoder_inputs = block['enc_self_attention'](
                    encoder_inputs,
                    encoder_inputs,
                    query_source_dist=encoder_time_dist,
                    bias=attention_bias,
                    training=training)
                encoder_inputs = block['enc_feed_forward'](
                    encoder_inputs, training=training)
        return self.output_normalization(encoder_inputs)


class DKTTLight(tf.keras.Model):
    def __init__(self, params, name='dktt_light'):
        super(DKTTLight, self).__init__(name=name)
        self.params = params
        self.embedding_layer = EmbeddingSharedWeights(
            hidden_size=params['hidden_size'],
            q_matrix=params['q_matrix'],
            q_matrix_trainable=params['q_matrix_trainable'],
            confidence=params['confidence'],
            temperature=params['temperature']
        )

        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)
        if params['item_difficulty']:
            self.difficulty = tf.keras.layers.Embedding(params['q_matrix'].shape[0], 1)

    def get_config(self):
        return {'params': self.params}

    def call(self, inputs, training=False):
        # encoder bias is used for enc_self_attn, enc_dec_attn; while decoder bias is used for dec_self_attn
        enc_self_attn_bias, enc_dec_attn_bias, dec_self_attn_bias = self.get_attention_bias(inputs)

        # adjust time unit
        encoder_times = tf.cast(inputs['encoder_times'], dtype=tf.float32) / self.params['time_unit']
        decoder_times = tf.cast(inputs['decoder_times'], dtype=tf.float32) / self.params['time_unit']

        # encoder inputs: shape = [batch_size, seq_len, hidden_size]
        # combine feature inputs
        encoder_inputs = self.embedding_layer({'items': inputs['encoder_items']})

        # time distance
        encoder_self_time_distance = self._get_time_dist(
            encoder_times,
            encoder_times)

        encoder_decoder_time_distance = self._get_time_dist(
            encoder_times,
            decoder_times)

        decoder_self_time_distance = self._get_time_dist(
            decoder_times,
            decoder_times)

        # encoder stack
        encoder_outputs = self.encode(
            encoder_inputs,
            encoder_self_time_distance,
            enc_self_attn_bias,
            training=training)

        # decoder stack
        logits = self.decode(
            inputs['decoder_items'],
            encoder_outputs,
            decoder_self_time_distance,
            encoder_decoder_time_distance,
            dec_self_attn_bias,
            enc_dec_attn_bias,
            training=training)

        if self.params['item_difficulty']:
            item_difficulty = self.difficulty(inputs['decoder_items'])
            logits -= item_difficulty[:, :, 0]

        return tf.nn.sigmoid(logits)

    def encode(self, encoder_inputs, encoder_self_time_distance, enc_self_attn_bias, training):
        if training:
            encoder_inputs = tf.nn.dropout(
                encoder_inputs, rate=self.params['layer_postprocess_dropout'])

        return self.encoder_stack(
            encoder_inputs,
            encoder_self_time_distance,
            enc_self_attn_bias,
            training=training)

    def decode(
            self,
            decoder_items,
            encoder_outputs,
            decoder_self_time_distance,
            encoder_decoder_time_distance,
            dec_self_attn_bias,
            enc_dec_attn_bias,
            training):
        # decoder_items = [(q_{t + 1}, 2)], where 2 = correct
        decoder_inputs = self.embedding_layer({'items': decoder_items})

        if training:
            decoder_inputs = tf.nn.dropout(
                decoder_inputs, rate=self.params['layer_postprocess_dropout'])

        decoder_outputs = self.decoder_stack(
            decoder_inputs,
            encoder_outputs,
            decoder_self_time_distance,
            encoder_decoder_time_distance,  # dec_time - enc_time
            dec_self_attn_bias,
            enc_dec_attn_bias,  # this is for enc_dec_attention_layer
            training=training)

        logits = self.embedding_layer(
            {'states': decoder_outputs,  # decoder_outputs,
             'query_items': decoder_items},
            mode='linear')  # shape = [batch_size, seq_len]

        return logits

    def get_attention_bias(self, inputs):
        """ get attention bias
        Parameters
        ----------
            inputs : Dict[tf.Tensor]
                a dictionary containing the following entries:
                'encoder_items','encoder_times', 'decoder_items', and
                'decoder_times'

        Returns
        -------
            enc_self_attn: tf.Tensor
            enc_dec_attn: tf.Tensor
            dec_self_attn: tf.Tensor
                all attention is a tf.Tensor with
                shape = [batch_size, 1, seq_len, seq_len],
                where 1 is reserved to broadcast to num_heads
        """

        # 1. mask out padding
        # bias shape = [batch_size, 1, 1, seq_len]
        encoder_padding_bias = get_padding_bias(inputs['encoder_items'])
        decoder_padding_bias = get_padding_bias(inputs['decoder_items'])

        # 2. mask out based on positions
        # bias shape = [1, 1, seq_len, seq_len] or [batch_size, 1, seq_len, seq_len]
        seq_len = tf.shape(inputs['encoder_items'])[1]
        if self.params['mask_out'] == 'future':
            # attn only self and past
            # this assumes there is at least 1 lag between encoder and decoder positions
            # e.g.,
            # encoder items: [x_0, x_1, x_2, ..., x_{n - 1}]
            # decoder items: [x_1, x_2, x_3, ..., x_{n}]
            # so:
            # 1. decoder at pos t can att to encoder at pos t without att to present and future
            # 2. encoder at pos t can att to encoder at pos 0 - t without att to future
            enc_self_attn_mask = get_decoder_self_attention_bias(seq_len)
            dec_self_attn_mask = enc_dec_attn_mask = enc_self_attn_mask
        elif self.params['mask_out'] == 'time_block':
            # in encoder: attn prev and curr time blocks
            # in decoder: attn prev blocks
            # useful in cases the inputs are a sequence of courses in semester blocks
            # e.g.,
            # encoder items: [0, x0, x1, x2], encoder blocks: [0, 1, 1, 2]
            # decoder items: [x0, x1, x2, x3], decoder blocks: [1, 1, 2, 2]
            # enc_self_attn_bias: x0 can attn to x1, even x1 is after x0
            # enc_dec_attn_bias: x3 can only attn to x0, x1, not x2. because only x0 and x1 are in prev blocks
            # dec_self_attn_bias: x3 can attn to x0, x1, x2 (in decoder inputs)
            enc_self_attn_mask = get_block_attention_bias(
                inputs['encoder_times'],
                inputs['encoder_times'],
                allow_current=True,
                dtype=encoder_padding_bias.dtype)
            enc_dec_attn_mask = get_block_attention_bias(
                inputs['encoder_times'],
                inputs['decoder_times'],
                allow_current=False,
                dtype=encoder_padding_bias.dtype)
            dec_self_attn_mask = get_block_attention_bias(
                inputs['decoder_times'],
                inputs['decoder_times'],
                allow_current=True,
                dtype=encoder_padding_bias.dtype)
        else:
            raise NotImplementedError

        enc_self_attn_bias = tf.math.minimum(encoder_padding_bias, enc_self_attn_mask)
        enc_dec_attn_bias = tf.math.minimum(encoder_padding_bias, enc_dec_attn_mask)
        dec_self_attn_bias = tf.math.minimum(decoder_padding_bias, dec_self_attn_mask)

        return enc_self_attn_bias, enc_dec_attn_bias, dec_self_attn_bias

    def _get_time_dist(self, tx, ty):
        # ty > tx
        ty = tf.expand_dims(ty, axis=-1)
        tx = tf.expand_dims(tx, axis=-2)

        t_y_x_dist = ty - tx
        mask = tf.cast(tf.greater_equal(t_y_x_dist, 0.), dtype=t_y_x_dist.dtype)
        t_y_x_dist *= mask

        return tf.nn.tanh(t_y_x_dist)
