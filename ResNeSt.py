import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

class BasicConvBlock(Model):
    def __init__(self, channel, k_size, strides=1, padding='same', group=1, use_bias=False):
        super(BasicConvBlock, self).__init__()
        
        self.block = tf.keras.Sequential([
            layers.Conv2D(channel, k_size, strides=strides, padding=padding, groups=group, use_bias=use_bias),
            layers.BatchNormalization(epsilon=1e-12),
            layers.ELU()
        ])
        
    def call(self, x, training=True):
        output = self.block(x, training=training)
        
        return output

class CardinalBlock(Model):
    def __init__(self, in_channel, out_channel, k_size, strides, radix, group):
        super(CardinalBlock, self).__init__()
        self.in_channel, self.out_channel, self.k_size, self.strides = in_channel, out_channel, k_size, strides
        self.radix, self.group = radix, group
        
        self.inter_channel = max(in_channel * radix // 4, 32)
        
        self.conv_block_1 = BasicConvBlock(out_channel * radix, k_size, strides=strides, group=group*radix)
        
        self.ga_pool_1 = layers.GlobalAveragePooling2D()
        
        self.conv_block_2 = BasicConvBlock(self.inter_channel, 1, group=group)
        
        self.conv_1 = layers.Conv2D(out_channel * radix, 1, strides=1, padding='same', groups=group, use_bias=False)
        
    def r_softmax(self, x):
        if self.radix > 1:
            out = layers.Reshape([self.group, self.radix, self.out_channel // self.group])(x)
            out = layers.Permute([2, 1, 3])(out)
            out = layers.Activation(tf.keras.activations.softmax)(out)
            out = layers.Reshape([1, 1, self.radix * self.out_channel])(out)
        else:
            out = layers.Activation(tf.keras.activations.sigmoid)(x)
            
        return out
        
    def call(self, x, training=True):        
        h_f = self.conv_block_1(x, training=training)
        
        h_f_shape = h_f.get_shape().as_list()
        
        split_hf = tf.split(h_f, num_or_size_splits=self.radix, axis=-1)
        sum_hf = layers.Add()(split_hf)
        
        gap = self.ga_pool_1(sum_hf, training=training)
        gap = layers.Reshape([1, 1, self.out_channel])(gap)
        gap = self.conv_block_2(gap, training=training)
        
        atten = self.conv_1(gap, training=training)
        atten = self.r_softmax(atten)
        
        atten_shape = atten.get_shape().as_list()
        split_atten = tf.split(atten, num_or_size_splits=self.radix, axis=-1)
        
        output = layers.Add()([a * s for a, s in zip(split_atten, split_hf)])
        
        return output

class ResidualBlock(Model):
    def __init__(self, in_channel, out_channel, k_size, strides, radix, group, b_width, channel_diff):
        super(ResidualBlock, self).__init__()
        
        self.in_channel, self.out_channel, self.k_size, self.strides = in_channel, out_channel, k_size, strides
        self.radix, self.group, self.b_width, self.channel_diff = radix, group, b_width, channel_diff
        expansion = 4
        
        self.group_width = int(out_channel * (b_width / 64.)) * group
        
        self.conv_block_1 = BasicConvBlock(self.group_width, 1)
        
        self.card_block_1 = CardinalBlock(self.group_width, self.group_width, k_size, strides, radix, group)
        
        self.conv_block_2 = tf.keras.Sequential([
            layers.Conv2D(out_channel * expansion, 1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(epsilon=1e-12)
        ])
        
        self.downsample = None
        if strides != 1 or channel_diff:
            self.downsample = tf.keras.Sequential([
                layers.Conv2D(out_channel * expansion, 1, strides=strides, padding='same', use_bias=False),
                layers.BatchNormalization(epsilon=1e-12)
            ])
        
    def call(self, x, training=True):
        residual = x
        
        h_f = self.conv_block_1(x, training=training)
        h_f = self.card_block_1(h_f, training=training)
        h_f = self.conv_block_2(h_f, training=training)
        
        if self.downsample is not None:
            residual = self.downsample(x, training=training)
            
        out = h_f + residual
        output = tf.nn.elu(out) 
        
        return output

class ResNeSt(Model):
    # Resnet18 like architecture
    def __init__(self, num_class=2, k_size=3, radix=2, group=2, b_width=64, deep_stem=False):
        super(ResNeSt, self).__init__()
        
        self.num_class = num_class
        self.k_size, self.radix, self.group, self.b_width = k_size, radix, group, b_width
        self.stem_width = 64
        self.in_channel = self.stem_width * 2 if deep_stem else 64
        
        # Deep Stem
        if deep_stem:
            self.conv_block_1 = tf.keras.Sequential([
                BasicConvBlock(self.stem_width, 3, strides=2),
                BasicConvBlock(self.stem_width, 3),
                BasicConvBlock(self.stem_width * 2, 3)
            ])
        else:
            self.conv_block_1 = BasicConvBlock(64, 7, strides=2)
        
        self.max_pool_1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid') # 32 x 32
        
        self.conv_block_2_1 = ResidualBlock(64, 64, k_size, 1, radix, group, b_width, channel_diff=True)
        self.conv_block_2_2 = ResidualBlock(64, 64, k_size, 1, radix, group, b_width, channel_diff=False)
        
        self.conv_block_3_1 = ResidualBlock(64, 128, k_size, 2, radix, group, b_width, channel_diff=True) # 16 x 16
        self.conv_block_3_2 = ResidualBlock(128, 128, k_size, 1, radix, group, b_width, channel_diff=False)
        
        self.conv_block_4_1 = ResidualBlock(128, 256, k_size, 2, radix, group, b_width, channel_diff=True) # 8 x 8
        self.conv_block_4_2 = ResidualBlock(256, 256, k_size, 1, radix, group, b_width, channel_diff=False)
        
        self.flatten = layers.Flatten()
        self.avg_pool_1 = layers.GlobalAveragePooling2D()
        
        self.dense_1 = layers.Dense(num_class)
        
    def call(self, x, training=True):
        h_f = self.conv_block_1(x, training=training)
        h_f = self.max_pool_1(h_f, training=training)
        
        # block 64
        h_f = self.conv_block_2_1(h_f, training=training)
        h_f = self.conv_block_2_2(h_f, training=training)
        # block 128
        h_f = self.conv_block_3_1(h_f, training=training)
        h_f = self.conv_block_3_2(h_f, training=training)
        # block 256 
        h_f = self.conv_block_4_1(h_f, training=training)
        h_f = self.conv_block_4_2(h_f, training=training)
        
        h_f = self.avg_pool_1(h_f, training=training)
        h_f_flatten = self.flatten(h_f, training=training)
        
        output = self.dense_1(h_f_flatten, training=training)
        
        return output

if __name__ == '__main__':
    batch_size = 32
    height = 64
    width = 64
    channel = 3

    test_data = tf.random.normal([batch_size, height, width, channel])

    model = ResNeSt()

    test_output = model(test_data, training=False)

    print(test_output.shape)
    print('Done!')
