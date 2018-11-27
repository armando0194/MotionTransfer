import argparse
import random
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import os
import math
import collections
import time

class Pix2PixModel:
    CROP_SIZE = 256
    EPS = 1e-12
    Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
    Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

    def __init__(self, args):
        self.args = args
        self.seed = random.randint(0, 2**31 - 1) if args.seed is None else args.seed

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Create output directory
        self.input_dir = Path(args.target_path)
        self.input_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(args.p2p_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(str(self.output_dir.joinpath("options.json")), "w") as f:
             f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    def get_name(self, path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name
    
    def preprocess(self, image):
        with tf.name_scope("preprocess"):
            # [0, 1] => [-1, 1]
            return image * 2 - 1

    def transform(self, image, seed):
        r = image
        if self.args.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces self.args nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [self.args.scale_size, self.args.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, self.args.scale_size - self.CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if self.args.scale_size > self.CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], self.CROP_SIZE, self.CROP_SIZE)
        elif self.args.scale_size < self.CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    def load_examples(self):
        train_dir = self.input_dir.joinpath('train_images')

        files = [str(x) for x in train_dir.iterdir() if x.is_file()]

        decode = tf.image.decode_png
        input_paths = sorted(files, key=lambda path: int(self.get_name(path)))
       
        # print(input_paths)
        with tf.name_scope("load_images"):
            path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
            # tf.data.Dataset.from_tensor_slices(input_paths).shuffle(tf.shape(input_tensor, out_type=tf.int64)[0]).
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)
            raw_input = decode(contents)
            raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
            
            with tf.control_dependencies([assertion]):
                raw_input = tf.identity(raw_input)

            raw_input.set_shape([None, None, 3])

            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = self.preprocess(raw_input[:,:width//2,:])
            b_images = self.preprocess(raw_input[:,width//2:,:]) 
            
            inputs, targets = [a_images, b_images]
            seed = random.randint(0, 2**31 - 1)

            with tf.name_scope("input_images"):
                input_images = self.transform(inputs, seed)

            with tf.name_scope("target_images"):
                target_images = self.transform(targets, seed)

            paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=self.args.batch_size)
            steps_per_epoch = int(math.ceil(len(input_paths) / self.args.batch_size))

        return self.Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )
    
    def lrelu(self, x, a):
        with tf.name_scope("lrelu"):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: self.args*x/2 - self.args*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

    def gen_conv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if self.args.separable_conv:
            return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

    def gen_deconv(self, batch_input, out_channels):
        # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
        initializer = tf.random_normal_initializer(0, 0.02)
        if self.args.separable_conv:
            _b, h, w, _c = batch_input.shape
            resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
        else:
            return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


    def discrim_conv(self, batch_input, out_channels, stride):
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


    def batchnorm(self, inputs):
        return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


    def create_generator(self, generator_inputs, generator_outputs_channels):
        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = self.gen_conv(generator_inputs, self.args.ngf)
            layers.append(output)

        layer_specs = [
            self.args.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            self.args.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            self.args.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            self.args.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            self.args.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            self.args.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            self.args.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = self.lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = self.gen_conv(rectified, out_channels)
                output = self.batchnorm(convolved)
                layers.append(output)

        layer_specs = [
            (self.args.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (self.args.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (self.args.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (self.args.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (self.args.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (self.args.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (self.args.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = self.gen_deconv(rectified, out_channels)
                output = self.batchnorm(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = self.gen_deconv(rectified, generator_outputs_channels)
            output = tf.tanh(output)
            layers.append(output)

        return layers[-1]

    
    def create_model(self, inputs, targets):
        def create_discriminator(discrim_inputs, discrim_targets):
            n_layers = 3
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope("layer_1"):
                convolved = self.discrim_conv(input, self.args.ndf, stride=2)
                rectified = self.lrelu(convolved, 0.2)
                layers.append(rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
                with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                    out_channels = self.args.ndf * min(2**(i+1), 8)
                    stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                    convolved = self.discrim_conv(layers[-1], out_channels, stride=stride)
                    normalized = self.batchnorm(convolved)
                    rectified = self.lrelu(normalized, 0.2)
                    layers.append(rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                convolved = self.discrim_conv(rectified, out_channels=1, stride=1)
                output = tf.sigmoid(convolved)
                layers.append(output)

            return layers[-1]

        with tf.variable_scope("generator"):
            out_channels = int(targets.get_shape()[-1])
            outputs = self.create_generator(inputs, out_channels)

        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_real = create_discriminator(inputs, targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                predict_fake = create_discriminator(inputs, outputs)

        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + self.EPS) + tf.log(1 - predict_fake + self.EPS)))

        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + self.EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
            gen_loss = gen_loss_GAN * self.args.gan_weight + gen_loss_L1 * self.args.l1_weight

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.args.lr, self.args.beta1)
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.args.lr, self.args.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

        global_step = tf.train.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        return self.Model(
            predict_real=predict_real,
            predict_fake=predict_fake,
            discrim_loss=ema.average(discrim_loss),
            discrim_grads_and_vars=discrim_grads_and_vars,
            gen_loss_GAN=ema.average(gen_loss_GAN),
            gen_loss_L1=ema.average(gen_loss_L1),
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=outputs,
            train=tf.group(update_losses, incr_global_step, gen_train),
        )


    def preprocess_lab(self, lab):
        with tf.name_scope("preprocess_lab"):
            L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
            # L_chan: black and white with input range [0, 100]
            # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
            # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
            return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

    def check_image(self, image):
        assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
        with tf.control_dependencies([assertion]):
            image = tf.identity(image)

        if image.get_shape().ndims not in (3, 4):
            raise ValueError("image must be either 3 or 4 dimensions")

        # make the last dimension 3 so that you can unstack the colors
        shape = list(image.get_shape())
        shape[-1] = 3
        image.set_shape(shape)
        return image

    def deprocess_lab(self, L_chan, a_chan, b_chan):
        with tf.name_scope("deprocess_lab"):
            # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
            return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)

    def lab_to_rgb(self, lab):
        with tf.name_scope("lab_to_rgb"):
            lab = self.check_image(lab)
            lab_pixels = tf.reshape(lab, [-1, 3])

            # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
            with tf.name_scope("cielab_to_xyz"):
                # convert to fxfyfz
                lab_to_fxfyfz = tf.constant([
                    #   fx      fy        fz
                    [1/116.0, 1/116.0,  1/116.0], # l
                    [1/500.0,     0.0,      0.0], # a
                    [    0.0,     0.0, -1/200.0], # b
                ])
                fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

                # convert to xyz
                epsilon = 6/29
                linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
                exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
                xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

                # denormalize for D65 white point
                xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

            with tf.name_scope("xyz_to_srgb"):
                xyz_to_rgb = tf.constant([
                    #     r           g          b
                    [ 3.2404542, -0.9692660,  0.0556434], # x
                    [-1.5371385,  1.8760108, -0.2040259], # y
                    [-0.4985314,  0.0415560,  1.0572252], # z
                ])
                rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
                # avoid a slightly negative number messing up the conversion
                rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
                linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
                exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
                srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

            return tf.reshape(srgb_pixels, tf.shape(lab))



    def augment(self, image, brightness):
        # (a, b) color channels, combine with L channel and convert to rgb
        a_chan, b_chan = tf.unstack(image, axis=3)
        L_chan = tf.squeeze(brightness, axis=3)
        lab = self.deprocess_lab(L_chan, a_chan, b_chan)
        rgb = self.lab_to_rgb(lab)
        return rgb

    def convert(self, image):
        if self.args.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [self.CROP_SIZE, int(round(self.CROP_SIZE * self.args.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)
    
    def deprocess(self, image):
        with tf.name_scope("deprocess"):
            # [-1, 1] => [0, 1]
            return (image + 1) / 2

    def save_images(self, fetches, step=None):
        image_dir = os.path.join(self.output_dir, "images")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        filesets = []
        for i, in_path in enumerate(fetches["paths"]):
            name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
            fileset = {"name": name, "step": step}
            for kind in ["inputs", "outputs", "targets"]:
                filename = name + "-" + kind + ".png"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
                fileset[kind] = filename
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i]
                with open(out_path, "wb") as f:
                    f.write(contents)
            filesets.append(fileset)
        return filesets

    def append_index(self, filesets, step=False):
        index_path = os.path.join(self.output_dir, "index.html")
        if os.path.exists(index_path):
            index = open(index_path, "a")
        else:
            index = open(index_path, "w")
            index.write("<html><body><table><tr>")
            if step:
                index.write("<th>step</th>")
            index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

        for fileset in filesets:
            index.write("<tr>")

            if step:
                index.write("<td>%d</td>" % fileset["step"])
            index.write("<td>%s</td>" % fileset["name"])

            for kind in ["inputs", "outputs", "targets"]:
                index.write("<td><img src='images/%s'></td>" % fileset[kind])

            index.write("</tr>")
        return index_path


    def train(self):
        examples = self.load_examples()
        print("examples count = %d" % examples.count)
        
        # inputs and targets are [batch_size, height, width, channels]
        model = self.create_model(examples.inputs, examples.targets)

        # undo colorization splitting on images that we use for display/output
        if self.args.lab_colorization:
            if self.args.which_direction == "AtoB":
                # inputs is brightness, this will be handled fine as self.args grayscale image
                # need to augment targets and outputs with brightness
                targets = self.augment(examples.targets, examples.inputs)
                outputs = self.augment(model.outputs, examples.inputs)
                # inputs can be deprocessed normally and handled as if they are single channel
                # grayscale images
                inputs = self.deprocess(examples.inputs)
            elif self.args.which_direction == "BtoA":
                # inputs will be color channels only, get brightness from targets
                inputs = self.augment(examples.inputs, examples.targets)
                targets = self.deprocess(examples.targets)
                outputs = self.deprocess(model.outputs)
            else:
                raise Exception("invalid direction")
        else:
            inputs = self.deprocess(examples.inputs)
            targets = self.deprocess(examples.targets)
            outputs = self.deprocess(model.outputs)

        # def convert(image):
        #     if self.args.aspect_ratio != 1.0:
        #         # upscale to correct aspect ratio
        #         size = [CROP_SIZE, int(round(CROP_SIZE * self.args.aspect_ratio))]
        #         image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        #     return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

        # reverse any processing on images so they can be written to disk or displayed to user
        with tf.name_scope("convert_inputs"):
            converted_inputs = self.convert(inputs)

        with tf.name_scope("convert_targets"):
            converted_targets = self.convert(targets)

        with tf.name_scope("convert_outputs"):
            converted_outputs = self.convert(outputs)

        with tf.name_scope("encode_images"):
            display_fetches = {
                "paths": examples.paths,
                "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
                "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
                "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            }

        # summaries
        with tf.name_scope("inputs_summary"):
            tf.summary.image("inputs", converted_inputs)

        with tf.name_scope("targets_summary"):
            tf.summary.image("targets", converted_targets)

        with tf.name_scope("outputs_summary"):
            tf.summary.image("outputs", converted_outputs)

        with tf.name_scope("predict_real_summary"):
            tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

        with tf.name_scope("predict_fake_summary"):
            tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

        tf.summary.scalar("discriminator_loss", model.discrim_loss)
        tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        saver = tf.train.Saver(max_to_keep=1)

        logdir = self.output_dir if (self.args.trace_freq > 0 or self.args.summary_freq > 0) else None
        sv = tf.train.Supervisor(logdir=str(logdir), save_summaries_secs=0, saver=None)
        with sv.managed_session() as sess:
            print("parameter_count =", sess.run(parameter_count))

            if self.args.checkpoint is not None:
                print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(self.args.checkpoint)
                saver.restore(sess, checkpoint)

            max_steps = 2**32
            if self.args.max_epochs is not None:
                max_steps = examples.steps_per_epoch * self.args.max_epochs
            if self.args.max_steps is not None:
                max_steps = self.args.max_steps

        #     if self.args.mode == "test":
        #         # testing
        #         # at most, process the test data once
        #         start = time.time()
        #         max_steps = min(examples.steps_per_epoch, max_steps)
        #         for step in range(max_steps):
        #             results = sess.run(display_fetches)
        #             filesets = save_images(results)
        #             for i, f in enumerate(filesets):
        #                 print("evaluated image", f["name"])
        #             index_path = append_index(filesets)
        #         print("wrote index at", index_path)
        #         print("rate", (time.time() - start) / max_steps)
        #     else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(self.args.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(self.args.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(self.args.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(self.args.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(self.args.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(self.args.display_freq):
                    print("saving display images")
                    filesets = self.save_images(results["display"], step=results["global_step"])
                    self.append_index(filesets, step=True)

                if should(self.args.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(self.args.progress_freq):
                    # global_step will have the correct step count if we resume from self.args checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * self.args.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * self.args.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(self.args.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(self.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break