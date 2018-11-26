from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
from post import *
import re

class OpenPoseModel:
    def __init__(self, mode, path):
        self.param ={'thre1':0.1, 'thre2':0.05, 'thre3':0.5}
        
        if mode == 'test':
            self.model = self.get_testing_model()
            self.model.load_weights(path)
        else:
            self.model = self.get_training_model()

    def relu(self, x): 
        return Activation('relu')(x)

    def conv(self, x, nf, ks, name, weight_decay):
        kernel_reg = l2(weight_decay[0]) if weight_decay else None
        bias_reg = l2(weight_decay[1]) if weight_decay else None

        x = Conv2D(nf, (ks, ks), padding='same', name=name,
                kernel_regularizer=kernel_reg,
                bias_regularizer=bias_reg,
                kernel_initializer=random_normal(stddev=0.01),
                bias_initializer=constant(0.0))(x)
        return x

    def pooling(self, x, ks, st, name):
        x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
        return x

    def vgg_block(self, x, weight_decay):
        # Block 1
        x = self.conv(x, 64, 3, "conv1_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 64, 3, "conv1_2", (weight_decay, 0))
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool1_1")

        # Block 2
        x = self.conv(x, 128, 3, "conv2_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv2_2", (weight_decay, 0))
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool2_1")

        # Block 3
        x = self.conv(x, 256, 3, "conv3_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 256, 3, "conv3_2", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 256, 3, "conv3_3", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 256, 3, "conv3_4", (weight_decay, 0))
        x = self.relu(x)
        x = self.pooling(x, 2, 2, "pool3_1")

        # Block 4
        x = self.conv(x, 512, 3, "conv4_1", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 512, 3, "conv4_2", (weight_decay, 0))
        x = self.relu(x)

        # Additional non vgg layers
        x = self.conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
        x = self.relu(x)

        return x


    def stage1_block(self, x, num_p, branch, weight_decay):
        # Block 1
        x = self.conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))

        return x


    def stageT_block(self, x, num_p, stage, branch, weight_decay):
        # Block 1
        x = self.conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
        x = self.relu(x)
        x = self.conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))

        return x


    def apply_mask(self, x, mask1, mask2, num_p, stage, branch, np_branch1, np_branch2):
        w_name = "weight_stage%d_L%d" % (stage, branch)

        # TODO: we have branch number here why we made so strange check
        assert np_branch1 != np_branch2 # we selecting branches by number of pafs, if they accidentally became the same it will be disaster

        if num_p == np_branch1:
            w = Multiply(name=w_name)([x, mask1])  # vec_weight
        elif num_p == np_branch2:
            w = Multiply(name=w_name)([x, mask2])  # vec_heat
        else:
            assert False, "wrong number of layers num_p=%d " % num_p
        return w


    def get_training_model(self, weight_decay=5e-4, np_branch1=38, np_branch2=19, stages = 6, gpus = None):

        img_input_shape = (None, None, 3)
        vec_input_shape = (None, None, np_branch1)
        heat_input_shape = (None, None, np_branch2)

        inputs = []
        outputs = []

        img_input = Input(shape=img_input_shape)
        vec_weight_input = Input(shape=vec_input_shape)
        heat_weight_input = Input(shape=heat_input_shape)

        inputs.append(img_input)
        if np_branch1 > 0:
            inputs.append(vec_weight_input)

        if np_branch2 > 0:
            inputs.append(heat_weight_input)

        #img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]
        img_normalized = img_input # will be done on augmentation stage

        # VGG
        stage0_out = self.vgg_block(img_normalized, weight_decay)

        # stage 1 - branch 1 (PAF)
        new_x = []
        if np_branch1 > 0:
            stage1_branch1_out = self.stage1_block(stage0_out, np_branch1, 1, weight_decay)
            w1 = self.apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1, np_branch1, np_branch2)
            outputs.append(w1)
            new_x.append(stage1_branch1_out)

        # stage 1 - branch 2 (confidence maps)

        if np_branch2 > 0:
            stage1_branch2_out = self.stage1_block(stage0_out, np_branch2, 2, weight_decay)
            w2 = self.apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2, np_branch1, np_branch2)
            outputs.append(w2)
            new_x.append(stage1_branch2_out)

        new_x.append(stage0_out)

        x = Concatenate()(new_x)

        # stage sn >= 2
        for sn in range(2, stages + 1):

            new_x = []
            # stage SN - branch 1 (PAF)
            if np_branch1 > 0:
                stageT_branch1_out = self.stageT_block(x, np_branch1, sn, 1, weight_decay)
                w1 = self.apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1, np_branch1, np_branch2)
                outputs.append(w1)
                new_x.append(stageT_branch1_out)

            # stage SN - branch 2 (confidence maps)
            if np_branch2 > 0:
                stageT_branch2_out = self.stageT_block(x, np_branch2, sn, 2, weight_decay)
                w2 = self.apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2, np_branch1, np_branch2)
                outputs.append(w2)
                new_x.append(stageT_branch2_out)

            new_x.append(stage0_out)

            if sn < stages:
                x = Concatenate()(new_x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def get_lrmult(self, model):

        # setup lr multipliers for conv layers
        lr_mult = dict()

        for layer in model.layers:

            if isinstance(layer, Conv2D):

                # stage = 1
                if re.match("Mconv\d_stage1.*", layer.name):
                    kernel_name = layer.weights[0].name
                    bias_name = layer.weights[1].name
                    lr_mult[kernel_name] = 1
                    lr_mult[bias_name] = 2

                # stage > 1
                elif re.match("Mconv\d_stage.*", layer.name):
                    kernel_name = layer.weights[0].name
                    bias_name = layer.weights[1].name
                    lr_mult[kernel_name] = 4
                    lr_mult[bias_name] = 8

                # vgg
                else:
                    print("matched as vgg layer", layer.name)
                    kernel_name = layer.weights[0].name
                    bias_name = layer.weights[1].name
                    lr_mult[kernel_name] = 1
                    lr_mult[bias_name] = 2

        return lr_mult


    def get_testing_model(self, np_branch1=38, np_branch2=19, stages = 6):

        img_input_shape = (None, None, 3)

        img_input = Input(shape=img_input_shape)

        img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

        # VGG
        stage0_out = self.vgg_block(img_normalized, None)

        stages_out = []

        # stage 1 - branch 1 (PAF)
        if np_branch1 > 0:
            stage1_branch1_out = self.stage1_block(stage0_out, np_branch1, 1, None)
            stages_out.append(stage1_branch1_out)

        # stage 1 - branch 2 (confidence maps)
        if np_branch2 > 0:
            stage1_branch2_out = self.stage1_block(stage0_out, np_branch2, 2, None)
            stages_out.append(stage1_branch2_out)

        x = Concatenate()(stages_out + [stage0_out])

        # stage t >= 2
        stageT_branch1_out = None
        stageT_branch2_out = None
        for sn in range(2, stages + 1):

            stages_out = []

            if np_branch1 > 0:
                stageT_branch1_out = self.stageT_block(x, np_branch1, sn, 1, None)
                stages_out.append(stageT_branch1_out)
            if np_branch2 > 0:
                stageT_branch2_out = self.stageT_block(x, np_branch2, sn, 2, None)
                stages_out.append(stageT_branch2_out)

            if sn < stages:
                x = Concatenate()(stages_out + [stage0_out])

        model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

        return model
    
    def predict(self, img):
        multiplier = [x * 368 / img.shape[0] for x in [0.5, 1.0, 1.5, 2.0]]
        
        heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
        paf_avg = np.zeros((img.shape[0], img.shape[1], 38))
        
        for m in range(len(multiplier)):
            scale = multiplier[m]

            imageToTest = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            imageToTest_padded, pad = padRightDownCorner(imageToTest, 8, 8)

            input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels)
            
            output_blobs = self.model.predict(input_img)

            # extract outputs, resize, and remove padding
            heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
            heatmap = cv2.resize(heatmap, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],:]
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

            paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
            paf = cv2.resize(paf, (0, 0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
            paf = cv2.resize(paf, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            paf_avg = paf_avg + paf / len(multiplier)

        return self.get_pose(self.param, heatmap_avg, paf_avg)
    
    def get_pose(self, param, heatmaps, pafs):
        shape = heatmaps.shape[:2]
        joint_list_per_joint_type = NMS(param, heatmaps)
        joint_list = np.array([tuple(peak) + (joint_type,) for joint_type,
                            joint_peaks in enumerate(joint_list_per_joint_type) for peak in joint_peaks])

        paf_upsamp = cv2.resize(pafs, shape, interpolation=cv2.INTER_CUBIC)
        connected_limbs = find_connected_joints(param, paf_upsamp, joint_list_per_joint_type)
        person_to_joint_assoc = group_limbs_of_same_person(connected_limbs, joint_list)

        return self.create_label(shape, joint_list, person_to_joint_assoc)
    
    def create_label(self, shape, joint_list, person_to_joint_assoc):
        label = np.zeros(shape, dtype=np.uint8)
        for limb_type in range(17):
            for person_joint_info in person_to_joint_assoc:
                joint_indices = person_joint_info[joint_to_limb_heatmap_relationship[limb_type]].astype(int)
                if -1 in joint_indices:
                    continue
                joint_coords = joint_list[joint_indices, :2]
                coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
                limb_dir = joint_coords[0, :] - joint_coords[1, :]
                limb_length = np.linalg.norm(limb_dir)
                angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
                polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), 4), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(label, polygon, limb_type+1)
        return label

    