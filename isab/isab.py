import tensorflow as tf
from einops import rearrange, repeat

class Attention(tf.keras.layers.Layer):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super(Attention, self).__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = tf.keras.layers.Dense(inner_dim, input_dim=dim, use_bias = False)
        self.to_kv = tf.keras.layers.Dense(inner_dim * 2, input_dim=dim, use_bias = False)
        self.to_out = tf.keras.layers.Dense(dim, input_dim=inner_dim, use_bias = False)
    
    def call(self,
        inputs,
        context,
        mask = None):
        
        h, scale = self.heads, self.scale

        q = self.to_q(inputs)
        k, v = tf.split(self.to_kv(context), num_or_size_splits=2, axis=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if mask is not None:
            mask_value = -tf.experimental.numpy.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b 1 1 n')
            dots = tf.where(mask, mask_value, dots)

        attn = tf.nn.softmax(dots)
        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

class Isab(tf.keras.layers.Layer):
    def __init__(self, dim, heads = 8, num_latents = None):
        super(Isab, self).__init__()
        if num_latents is not None:
            self.latents = tf.Variable(tf.random.normal((num_latents, dim)))
        else:
            self.latents = None
        self.attn1 = Attention(dim, heads)
        self.attn2 = Attention(dim, heads)
    
    def call(self, inputs, latents = None, mask = None):
        b = inputs.shape[0]
        if latents is not None and self.latents is not None:
            assert 'you can only either learn the latents within the module, or pass it in externally'
        if latents is None:
            latents = self.latents
        if len(latents.shape) == 2:
            latents = repeat(latents, 'n d -> b n d', b = b)
        latents = self.attn1(latents, inputs, mask = mask)
        out = self.attn2(inputs, latents)
        return out, latents