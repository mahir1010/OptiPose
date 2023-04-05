import tensorflow as tf
from cvkit import MAGIC_NUMBER
from tensorflow.keras.layers import MultiHeadAttention, PReLU, Dense, Reshape, Input, Masking, Concatenate


def sub_context_model(cm_index, index, inputs, concat, key_dim=64, num_heads=1, dropout=0):
    x, attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, name=f'cm_{cm_index}_scm_{index}_attn',
                                 dropout=dropout)(inputs,
                                                  inputs,
                                                  return_attention_scores=True)
    x = tf.concat([x, concat], axis=-1)
    x = Dense(key_dim, activation=PReLU(), kernel_regularizer='l1_l2', name=f'cm_{cm_index}_scm_{index}_out')(x)
    return x


def context_model(index, inp, concat_inp, num_sub_ck, embedding_dims=30, num_heads=1):
    x = inp
    c = concat_inp
    for i in range(num_sub_ck):
        x = sub_context_model(index, i, x, c, num_heads=num_heads, key_dim=embedding_dims, dropout=0 if i == 0 else 0.2)
        c = x + concat_inp
    return x


def optipose_postural_autoencoder(window_size, n_parts, n_pcm, n_scm, multi_heads=1, weights=None):
    inputs = Input(shape=(window_size, n_parts, 3))
    inp = Reshape((-1, n_parts * 3))(inputs)
    inp = Masking(mask_value=0)(inp)
    concat_inp = tf.cast(inp > MAGIC_NUMBER, inp.dtype) * inp
    outputs = []
    for i in range(n_pcm):
        outputs.append(
            context_model(i, inp, concat_inp, n_scm, embedding_dims=n_parts * 3, num_heads=multi_heads))
    output = Concatenate(name='pcm_merge')(outputs)
    output = Dense(n_parts * 3, name='pcm_out', kernel_regularizer='l1_l2')(output)
    output = Reshape((window_size, n_parts, 3))(output)
    model = tf.keras.Model(inputs, output)
    if weights is not None:
        latest = tf.train.latest_checkpoint(weights)
        model.load_weights(latest).expect_partial()
    return model


if __name__ == '__main__':
    optipose_postural_autoencoder(30, 16, 10, 7, 1, 60).summary()
