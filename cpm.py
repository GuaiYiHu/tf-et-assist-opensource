import tensorflow as tf


class PafNet:
    def __init__(self, inputs_x, use_bn=False, mask_paf=None, mask_hm=None, gt_hm=None, gt_paf=None, stage_num=6,
                 hm_channel_num=19, paf_channel_num=38):
        self.inputs_x = inputs_x
        self.mask_paf = mask_paf
        self.mask_hm = mask_hm
        self.gt_hm = gt_hm
        self.gt_paf = gt_paf
        self.stage_num = stage_num
        self.paf_channel_num = paf_channel_num
        self.hm_channel_num = hm_channel_num
        self.use_bn = use_bn

    def add_layers(self, inputs):
        net = self.conv2(inputs=inputs, filters=256, padding='SAME', kernel_size=3, normalization=self.use_bn,
                         name='cpm_1')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn,
                         name='cpm_2')
        return net

    def stage_1(self, inputs, out_channel_num, name):
        net = self.conv2(inputs=inputs, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn,
                         name=name + '_conv1')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn,
                         name=name + '_conv2')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=3, normalization=self.use_bn,
                         name=name + '_conv3')
        net = self.conv2(inputs=net, filters=512, padding='SAME', kernel_size=1, normalization=self.use_bn,
                         name=name + '_conv4')
        net = self.conv2(inputs=net, filters=out_channel_num, padding='SAME', kernel_size=1, act=False,
                         normalization=self.use_bn, name=name + '_conv5')
        return net

    def stage_t(self, inputs, out_channel_num, name):
        net = self.conv2(inputs=inputs, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn,
                         name=name + '_conv1')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn,
                         name=name + '_conv2')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn,
                         name=name + '_conv3')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn,
                         name=name + '_conv4')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=7, normalization=self.use_bn,
                         name=name + '_conv5')
        net = self.conv2(inputs=net, filters=128, padding='SAME', kernel_size=1, normalization=self.use_bn,
                         name=name + '_conv6')
        net = self.conv2(inputs=net, filters=out_channel_num, padding='SAME', kernel_size=1, act=False,
                         name=name + '_conv7')
        return net

    def conv2(self, inputs, filters, padding, kernel_size, name, act=True, normalization=False):
        channels_in = inputs[0, 0, 0, :].get_shape().as_list()[0]
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, channels_in, filters], trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases', shape=[filters], trainable=True,
                                initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding=padding)
            output = tf.nn.bias_add(conv, b)
            if normalization:
                axis = list(range(len(output.get_shape()) - 1))
                mean, variance = tf.nn.moments(conv, axes=axis)
                scale = tf.Variable(tf.ones([filters]), name='scale')
                beta = tf.Variable(tf.zeros([filters]), name='beta')
                output = tf.nn.batch_normalization(output, mean, variance, offset=beta, scale=scale,
                                                   variance_epsilon=0.0001)
            if act:
                output = tf.nn.relu(output, name=scope.name)
        tf.summary.histogram('conv', conv)
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('output', output)

        return output

    def convlstm(self, name, input, state=None, filters=48, kernel_size=3, forget_bias=1.0, activation=tf.tanh,
                 peephole=False, normalize=False):
        kernel = [kernel_size, kernel_size]

        if state is None:
            x = input
        else:
            c, h = state
            x = tf.concat([input, h], axis=3)
        n = x.shape[-1].value
        m = 4 * filters if filters > 1 else 4
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('kernel', kernel + [n, m])
            y = tf.nn.convolution(x, W, 'SAME')
            if not normalize:
                y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
            j, i, f, o = tf.split(y, 4, axis=3)
            if peephole:
                i += tf.get_variable('W_ci', c.shape[1:]) * c
                f += tf.get_variable('W_cf', c.shape[1:]) * c

            if normalize:
                j = tf.contrib.layers.layer_norm(j)
                i = tf.contrib.layers.layer_norm(i)
                f = tf.contrib.layers.layer_norm(f)

            f = tf.sigmoid(f + forget_bias)
            i = tf.sigmoid(i)
            c = c * f + i * activation(j) if state is not None else i * activation(j)

            if peephole:
                o += tf.get_variable('W_co', c.shape[1:]) * c

            if normalize:
                o = tf.contrib.layers.layer_norm(o)
                c = tf.contrib.layers.layer_norm(c)

            o = tf.sigmoid(o)
            h = o * activation(c)

            state = (c, h)
        return h, state

    def gen_net(self):
        paf_pre = []
        hm_pre = []
        with tf.variable_scope('openpose_layers'):
            with tf.variable_scope('cpm_layers'):
                added_layers_out = self.add_layers(inputs=self.inputs_x)

            with tf.variable_scope('stage_1'):
                paf_net = self.stage_1(inputs=added_layers_out, out_channel_num=self.paf_channel_num, name='stage1_paf')
                hm_net = self.stage_1(inputs=added_layers_out, out_channel_num=self.hm_channel_num, name='stage1_hm')
                paf_pre.append(paf_net)
                hm_pre.append(hm_net)
                net = tf.concat([hm_net, paf_net, added_layers_out], 3)
                net, state = self.convlstm(input=net, filters=148, kernel_size=3, name='stage1_lstm')

            for i in range(self.stage_num - 1):
                with tf.variable_scope('stage_%d' % (i + 2)):
                    hm_net = self.stage_t(inputs=net, out_channel_num=self.hm_channel_num, name='stage%d_hm' % (i + 2))
                    paf_net = self.stage_t(inputs=net, out_channel_num=self.paf_channel_num,
                                           name='stage%d_paf' % (i + 2))
                    paf_pre.append(paf_net)
                    hm_pre.append(hm_net)
                    if i < self.stage_num - 2:
                        net = tf.concat([hm_net, paf_net, added_layers_out], 3)
                        net, state = self.convlstm(input=net, state=state, filters=148, kernel_size=3,
                                                   name='stage%d_lstm' % (i + 2))

        return hm_pre, paf_pre, added_layers_out

    def gen_hand_net(self):
        paf_pre = []
        hm_pre = []
        with tf.variable_scope('openpose_layers'):
            with tf.variable_scope('cpm_layers'):
                added_layers_out = self.add_layers(inputs=self.inputs_x)

            with tf.variable_scope('stage_1'):
                paf_net = self.stage_1(inputs=added_layers_out, out_channel_num=self.paf_channel_num, name='stage1_paf')
                hm_net = self.stage_1(inputs=added_layers_out, out_channel_num=self.hm_channel_num, name='stage1_hm')
                paf_pre.append(paf_net)
                hm_pre.append(hm_net)
                net = tf.concat([hm_net, paf_net, added_layers_out], 3)
                net, state = self.convlstm(input=net, filters=148, kernel_size=3, name='stage1_lstm')

            for i in range(self.stage_num - 1):
                with tf.variable_scope('stage_%d' % (i + 2)):
                    hm_net = self.stage_t(inputs=net, out_channel_num=self.hm_channel_num, name='stage%d_hm' % (i + 2))
                    paf_net = self.stage_t(inputs=net, out_channel_num=self.paf_channel_num,
                                           name='stage%d_paf' % (i + 2))
                    paf_pre.append(paf_net)
                    hm_pre.append(hm_net)
                    if i < self.stage_num - 2:
                        net = tf.concat([hm_net, paf_net, added_layers_out], 3)
                        net, state = self.convlstm(input=net, state=state, filters=148, kernel_size=3,
                                                   name='stage%d_lstm' % (i + 2))

        return hm_pre, paf_pre, added_layers_out
