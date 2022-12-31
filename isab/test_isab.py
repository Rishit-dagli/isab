import numpy as np
import tensorflow as tf

from isab.isab import Isab

class IsabTest(tf.test.TestCase):
    def setUp(self):
        super(IsabTest, self).setUp()

        self.attn = Isab(
            dim = 512,
            heads = 8,
            num_latents = 128
        )
        self.seq = tf.random.normal((1, 16384, 512)) # (batch, seq, dim)
        self.mask = tf.ones((1, 16384), dtype = tf.bool) # (batch, seq)

    def test_shape_and_rank(self):
        outputs = self.attn(self.seq, mask = self.mask)

        self.assertEqual(tf.rank(outputs[0]), 3)
        self.assertEqual(tf.rank(outputs[1]), 3)
        self.assertShapeEqual(np.zeros((1, 16384, 512)), outputs[0])
        self.assertShapeEqual(np.zeros((1, 128, 512)), outputs[1])


if __name__ == "__main__":
    tf.test.main()