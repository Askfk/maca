"""Build Caption Layer."""
import tensorflow as tf


class CaptionAttention(tf.keras.Model):
    def __init__(self, units, config, **kwargs):
        super(CaptionAttention, self).__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units, name='macacnn_W1')
        self.W2 = tf.keras.layers.Dense(units, name='macacnn_W2')
        self.V = tf.keras.layers.Dense(1, name='macacnn_V')
        self.config = config

    def call(self, inputs):
        # features(encoder output) shape == (batch_size, num_rois, pool_size^2(49), embedding_dim)
        features = inputs[0]
        # hidden shape == (batch_size, num_rois, units)
        hidden = inputs[1]

        # hidden_with_time_axis shape == (batch_size, num_rois, 1, units)
        hidden_with_time_axis = tf.expand_dims(hidden, 2)
        shape = hidden_with_time_axis.shape
        # hidden_with_time_axis = tf.reshape(hidden, [hidden.shape[0], -1, 1, hidden.shape[-1]])
        # score shape == (batch_size, num_rois, pool_size * pool_size, units)
        score = tf.nn.tanh(self.W1(features) + self.W2(tf.reshape(hidden_with_time_axis,
                                                                  [self.config.BATCH_SIZE, -1, 1, self.config.UNITS])))

        # attention_weights shape == (batch_size, num_rois, 49, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=2)

        # context_vector shape after sum == (batch_size, num_rois, 49, embedding_dim)
        context_vector = attention_weights * features
        # [batch, num_rois, embedding_dim]
        context_vector = tf.reduce_mean(context_vector, axis=2)

        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    # The encoder passes features through a fully connected layer
    def __init__(self, embedding_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # shape after fc == (batch_size, num_rois, pool_size * pool_size, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim, name='macacnn_encoder_fc')

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size, config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.units = units
        self.config = config

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name='macacnn_embedding')
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name='macacnn_gru')
        self.fc1 = tf.keras.layers.Dense(self.units, name='macacnn_decoder_fc1')
        self.fc2 = tf.keras.layers.Dense(vocab_size, name='macacnn_decoder_fc2')

        self.attention = CaptionAttention(self.units, config)

    def call(self, inputs):
        # x shape == (batch_size, num_rois, 1)
        x = inputs[0]
        features = inputs[1]
        hidden = inputs[2]

        # Defining attention as a separate model
        context_vector, attention_weights = self.attention([features, hidden])

        # x shape after passing through embedding == (batch_size, num_rois, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, num_rois, 1, embedding_dim * 2)
        x = tf.concat([tf.expand_dims(context_vector, 2), x], axis=-1)
        # x shape after squeeze == [batch, num_rois, embedding_dim + hidden_size]
        # x = tf.squeeze(x, axis=2)
        # Reshape x to fit gru input [batch * num_rois, 1, embedding_dim + hidden_size]
        x = tf.reshape(x, [-1, x.shape[2], self.config.EMBEDDING_DIM * 2])

        # Passing the concatenated vector to the GRU
        # output == [batch * num_rois, 1, embedding_dim * 2]
        # state == [batch * num_rois, embedding_dim * 2]
        output, state = self.gru(x)

        # reshape output to [batch, num_rois, embedding_dim * 2]
        # state to [batch, num_rois, embedding_dim + hidden_size]
        output = tf.reshape(output, [self.config.BATCH_SIZE, -1, output.shape[-1]])
        state = tf.reshape(state, [self.config.BATCH_SIZE, -1, state.shape[-1]])

        # shape == (batch_size, num_rois, self.units)
        x = self.fc1(output)

        # output shape == (batch_size, num_rois, vocab_size)
        logits = self.fc2(x)
        # [batch, num_rois, vocab_size]
        probs = tf.nn.softmax(logits, axis=2)

        return logits, probs, state, attention_weights

    def reset_state(self, rois):
        tile_nums = self.units // rois.shape[-1]
        hidden = tf.zeros_like(rois)
        hidden = tf.tile(hidden, [1, 1, tile_nums])
        return hidden


@tf.function
def reset_state(rois, units, tokenizer):
    hidden = tf.zeros_like(rois[:, :, 0, 0, 0])
    hidden = tf.expand_dims(hidden, 2)
    # [batch, num_rois, units]
    hidden = tf.tile(hidden, [1, 1, units])

    # [batch, num_rois]
    dec_input = tf.ones_like(rois[:, :, 0, 0, 0])
    # dec_input.shape = [batch_size, num_rois, 1]
    dec_input = tf.expand_dims(dec_input * tokenizer.word_index['<start>'], 2)

    return hidden, dec_input


def build_caption_layer_graph(rois, pool_size, tokenizer, mode, config, target_caption=None):
    """
    Builds the computational graph of caption
    :param target_caption: [batch, num_rois, MAX_LENGTH(12)]
    :param mode:
    :param tokenizer:
    :param pool_size:
    :param rois: [batch, num_rois, 7, 7, channels]
    :param config:
    :return:
    """

    assert mode in ['training', 'inference']

    # Reshape x
    # Shape: [batch, num_rois, pool_size * pool_size, channels]
    processed_x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [x.shape[0], -1, pool_size ** 2, x.shape[4]]),
                                         name='macacnn_caption_reshape_roialign')(rois)
    # print(processed_x.shape)

    encoder = Encoder(config.EMBEDDING_DIM, name='macacnn_encoder')
    decoder = Decoder(config.EMBEDDING_DIM,
                      config.UNITS,
                      config.VOCAB_SIZE,
                      config, name='macacnn_decoder')

    # print(dec_input.shape)
    # [batch, num_rois, 49, EMBEDDING_DIM(256)]
    features = encoder(processed_x)

    hidden, dec_input = reset_state(rois, config.UNITS, tokenizer)

    # # shape = [batch, num_rois, units]
    # tile_nums = config.UNITS // rois.shape[-1]
    # hidden = tf.zeros_like(rois)
    # hidden = tf.tile(hidden, [1, 1, tile_nums])
    # print('hidden + {}'.format(hidden.shape))

    # Initialize decoder
    # attention_model = CaptionAttention(config.UNITS)
    # embedding = tf.keras.layers.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)
    # gru = tf.keras.layers.GRU(config.UNITS,
    #                           return_sequences=True,
    #                           return_state=True,
    #                           recurrent_initializer='glorot_uniform')
    # fc1 = tf.keras.layers.Dense(config.UNITS)
    # fc2 = tf.keras.layers.Dense(config.VOCAB_SIZE)

    # outputs = np.zeros([roi_shape[0], roi_shape[1], config.MAX_LENGTH, config.VOCAB_SIZE])
    # outputs = tf.expand_dims(tf.zeros_like(rois), axis=3)
    # outputs = tf.tile(outputs, [1, 1, config.MAX_LENGTH // roi_shape[-1], config.VOCAB_SIZE])
    # outputs = outputs.eval()
    outputs = []
    if mode == 'training':
        for i in range(1, config.MAX_LENGTH):
            # dec_input = tf.expand_dims(target_caption[:, :, i], 2)
            # context_vector, attentions = attention_model([features, hidden])
            # x = embedding(dec_input)
            # x = tf.keras.layers.Lambda(lambda x: tf.concat([tf.expand_dims(context_vector, 2), x], axis=-1))(x)
            # # x = tf.concat([tf.expand_dims(context_vector, 2), x], axis=-1)
            # shape = x.shape
            # x = tf.keras.layers.Reshape([-1, shape[2], shape[3]])(x)
            # # x = tf.reshape(x, [-1, x.shape[2], x.shape[3]])
            # output, state = gru(x)
            # output = tf.keras.layers.Reshape((shape[0], -1, output.shape[-1]))(output)
            # # output = tf.reshape(output, [shape[0], -1, output.shape[-1]])
            # hidden = tf.keras.layers.Reshape((shape[0], -1, state.shape[-1]))(state)
            # # hidden = tf.reshape(state, [shape[0], -1, state.shape[-1]])
            # x = fc1(output)
            # # output shape == (batch_size, num_rois, vocab_size)
            # logits = fc2(x)
            # outputs.append(logits)
            # dec_input = tf.expand_dims(target_caption[:, :, i], 2)
            # print(i + 1)
            # Passing the features through the decoder
            predictions, _, hidden, _ = decoder([dec_input, features, hidden])
            # Use teacher forcing to train the model
            dec_input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, i], 2))(target_caption)
            #
            # print(predictions.shape)
            # print(hidden.shape)
            #
            outputs.append(predictions)
            # attentions = tf.squeeze(attentions, axis=3)
            # attention_weights[:, :, :, ]
            # [batch, num_rois, 1]

        # # shape == [MAX_LENGTH - 1, batch, num_rois, vocab_size]
        # outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=0),
        #                                  name='macacnn_outputs_stack')(outputs)
        # outputs = tf.stack(outputs, axis=0)
        # shape == [batch, num_rois, MAX_LENGTH - 1, vocab_size]
        outputs = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[1, 2, 0, 3]),
                                         name='macacnn_captions')(outputs)
        # outputs = tf.transpose(outputs, perm=[1, 2, 0, 3])
        # print('outputs.shape == {}'.format(outputs.shape))
        return outputs, None
    else:
        attention_weights = []
        for i in range(config.MAX_LENGTH):
            # # print(i)
            # context_vector, attentions = attention_model([features, hidden])
            # attentions = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(attentions)
            # attention_weights.append(attentions)
            # # attention_weights.append(tf.squeeze(attentions, axis=3))
            # x = embedding(dec_input)
            # x = tf.keras.layers.Lambda(lambda x: tf.concat([tf.expand_dims(context_vector, 2), x], axis=-1))(x)
            # # x = tf.concat([tf.expand_dims(context_vector, 2), x], axis=-1)
            # shape = x.shape
            # x = tf.keras.layers.Reshape([-1, shape[2], shape[3]])(x)
            # # x = tf.reshape(x, [-1, x.shape[2], x.shape[3]])
            # output, state = gru(x)
            # output = tf.keras.layers.Reshape([shape[0], -1, output.shape[-1]])(output)
            # # output = tf.reshape(output, [shape[0], -1, output.shape[-1]])
            # hidden = tf.keras.layers.Reshape([shape[0], -1, state.shape[-1]])(state)
            # # hidden = tf.reshape(state, [shape[0], -1, state.shape[-1]])
            # x = fc1(output)
            #
            # # output shape == (batch_size, num_rois, vocab_size)
            # logits = fc2(x)
            #
            # probs = tf.keras.layers.Activation('softmax')(logits)
            # # probs = tf.nn.softmax(logits, axis=2)
            # outputs.append(probs)
            # dec_input = tf.keras.layers.Lambda(
            #     lambda x: tf.expand_dims(tf.argmax(x, axis=2, output_type=tf.int32), 2))(probs)
            # # dec_input = tf.expand_dims(tf.argmax(probs, axis=2, output_type=tf.int32), 2)

            _, probs, hidden, attentions = decoder([dec_input, features, hidden])
            attentions = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3))(attentions)
            attention_weights.append(attentions)
            # print(i)
            # print(hidden.shape)
            # attention_weights.append(tf.squeeze(attentions, axis=3))
            outputs.append(probs)

            dec_input = tf.keras.layers.Lambda(
                lambda x: tf.expand_dims(tf.argmax(x, axis=2, output_type=tf.int32), 2))(probs)
            # print(dec_input.shape)
            # dec_input = tf.expand_dims(tf.argmax(probs, axis=2, output_type=tf.int32), 2)
            # print(dec_input.shape)

        # # shape == [MAX_LENGTH, batch, num_rois, vocab_size]
        # outputs = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=0),
        #                                  name='macacnn_outputs_stack')(outputs)
        # outputs = tf.stack(outputs, axis=0)
        # shape == [batch, num_rois, MAX_LENGTH, vocab_size]
        outputs = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[1, 2, 0, 3]),
                                         name='maca_captions')(outputs)
        # outputs = tf.transpose(outputs, perm=[1, 2, 0, 3])
        # shape == [MAX_LENGTH, batch, num_rois, pool * pool]
        # attention_weights = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=0),
        #                                            name='macacnn_attentions_stack')(attention_weights)
        # attention_weights = tf.stack(attention_weights, axis=0)
        # shape == [batch, num_rois, pool * pool, MAX_LENGTH]
        attention_weights = tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[1, 2, 3, 0]),
                                                   name='caption_attention_weights')(attention_weights)
        # attention_weights = tf.transpose(attention_weights, perm=[1, 2, 3, 0])
        # print('outputs.shape == {}'.format(outputs.shape))
        return outputs, attention_weights

