import tensorflow as tf
from tensorflow.keras import layers

# Positional Encoding Layer
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        return self.pos_emb(positions)

# Feed Forward Network
def feed_forward(d_model, d_ff):
    return tf.keras.Sequential([
        layers.Dense(d_ff, activation='relu'),
        layers.Dense(d_model)
    ])

# Encoder Layer
class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = feed_forward(d_model, d_ff)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn = self.mha(x, x, attention_mask=mask)
        out1 = self.norm1(x + self.dropout1(attn, training=training))
        ffn_out = self.ffn(out1)
        return self.norm2(out1 + self.dropout2(ffn_out, training=training))

# Decoder Layer
class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate):
        super().__init__()
        self.self_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.cross_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = feed_forward(d_model, d_ff)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.self_mha(x, x, attention_mask=look_ahead_mask)
        out1 = self.norm1(x + self.dropout1(attn1, training=training))

        attn2 = self.cross_mha(out1, enc_output, attention_mask=padding_mask)
        out2 = self.norm2(out1 + self.dropout2(attn2, training=training))

        ffn_out = self.ffn(out2)
        return self.norm3(out2 + self.dropout3(ffn_out, training=training))

# Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, num_heads, d_model, d_ff, input_vocab_size, target_vocab_size, max_len, dropout_rate=0.1):
        super().__init__()
        self.encoder_embed = layers.Embedding(input_vocab_size, d_model)
        self.decoder_embed = layers.Embedding(target_vocab_size, d_model)

        self.encoder_pos = PositionalEncoding(max_len, d_model)
        self.decoder_pos = PositionalEncoding(max_len, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]
        self.dec_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

        self.final_linear = layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder_embed(inp) + self.encoder_pos(inp)
        for layer in self.enc_layers:
            enc_output = layer(enc_output, training, enc_padding_mask)

        dec_output = self.decoder_embed(tar) + self.decoder_pos(tar)
        for layer in self.dec_layers:
            dec_output = layer(dec_output, enc_output, training, look_ahead_mask, dec_padding_mask)

        return self.final_linear(dec_output)

# Masking Utilities
def create_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

def create_combined_mask(tar):
    look_ahead = create_look_ahead_mask(tf.shape(tar)[1])
    padding = create_padding_mask(tar)
    return tf.maximum(look_ahead, padding)
