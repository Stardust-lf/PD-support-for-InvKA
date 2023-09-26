import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import numpy as np
tf.compat.v1.disable_eager_execution()
def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

model3DCNN = keras.Sequential([
    keras.layers.Conv3D(name='conv1',kernel_size=(7,7,7),strides=4,batch_size=1,filters=16),
    keras.layers.MaxPooling3D(pool_size=(3,3,3),strides=2),
    keras.layers.Conv3D(name='conv2',kernel_size=(5,5,5),strides=1,batch_size=1,filters=64),
    keras.layers.MaxPooling3D(pool_size=(3,3,3),strides=2),
    keras.layers.Conv3D(name='conv3',kernel_size=(3,3,3),strides=1,batch_size=1,filters=256),
    keras.layers.Flatten(),
    keras.layers.Dense(2)
])
class AlexNet8(Model):
    def __init__(self):
        super(AlexNet8, self).__init__()
        self.c1 = Conv2D(filters=96, kernel_size=(3, 3))
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c2 = Conv2D(filters=256, kernel_size=(3, 3))
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.c3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                         activation='relu')

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                         activation='relu')
        self.p3 = MaxPool2D(pool_size=(3, 3), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(2048, activation='relu')
        self.d1 = Dropout(0.5)
        self.f2 = Dense(2048, activation='relu')
        self.d2 = Dropout(0.5)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.c3(x)

        x = self.c4(x)

        x = self.c5(x)
        x = self.p3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d1(x)
        x = self.f2(x)
        x = self.d2(x)
        y = self.f3(x)
        return y
from tensorflow.keras import layers, models, Model, Sequential

def GoogLeNet(im_height=224, im_width=224, class_num=12, aux_logits=False):
    # tensorflow中的tensor通道排序是NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # (None, 224, 224, 3)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding="SAME", activation="relu", name="conv2d_1")(input_image)
    # (None, 112, 112, 64)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_1")(x)
    # (None, 56, 56, 64)
    x = layers.Conv2D(64, kernel_size=1, activation="relu", name="conv2d_2")(x)
    # (None, 56, 56, 64)
    x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
    # (None, 56, 56, 192)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)

    # (None, 28, 28, 192)
    x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    # (None, 28, 28, 256)
    x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)

    # (None, 28, 28, 480)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
    # (None, 14, 14, 480)
    x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)
    if aux_logits:#如果为true就创建辅助分类器
        aux1 = InceptionAux(class_num, name="aux_1")(x)

    # (None, 14, 14, 512)
    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    # (None, 14, 14, 512)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    # (None, 14, 14, 512)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
    if aux_logits:
        aux2 = InceptionAux(class_num, name="aux_2")(x)

    # (None, 14, 14, 528)
    x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    # (None, 14, 14, 532)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_4")(x)

    # (None, 7, 7, 832)
    x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    # (None, 7, 7, 832)
    x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
    # (None, 7, 7, 1024)
    x = layers.AvgPool2D(pool_size=7, strides=1, name="avgpool_1")(x)

    # (None, 1, 1, 1024)
    x = layers.Flatten(name="output_flatten")(x)
    # (None, 1024)
    x = layers.Dropout(rate=0.4, name="output_dropout")(x)
    x = layers.Dense(class_num, name="output_dense")(x)
    # (None, class_num)
    aux3 = layers.Softmax(name="aux_3")(x)

    if aux_logits:
        model = models.Model(inputs=input_image, outputs=[aux1, aux2, aux3])#第一个和第二个是辅助分类器 第三个是主输出
    else:
        model = models.Model(inputs=input_image, outputs=aux3)
    return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):#**kwargs传入我们的层名称
        super(Inception, self).__init__(**kwargs)
        self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, activation="relu")
        #有两个卷积层 用sequential类讲二者合并   步距是=1 就没有写 保证可以在深度上进行拼接
        self.branch2 = Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch3x3, kernel_size=3, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch3 = Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, activation="relu"),
            layers.Conv2D(ch5x5, kernel_size=5, padding="SAME", activation="relu")])      # output_size= input_size

        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),  # caution: default strides==pool_size
            layers.Conv2D(pool_proj, kernel_size=1, activation="relu")])                  # output_size= input_size

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])#将四个分支的输出在深度上进行拼接
        return outputs

#定义辅助分类器结构
class InceptionAux(layers.Layer):
    def __init__(self, num_classes, **kwargs):#方便定义层名称
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")
        self.fc1 = layers.Dense(1024, activation="relu")
        self.fc2 = layers.Dense(num_classes)
        self.softmax = layers.Softmax()#将输出值转化为概率分布

    def call(self, inputs, **kwargs):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14  N是batch
        x = self.averagePool(inputs)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 2048
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        x = self.softmax(x)

        return x
#model = GoogLeNet(im_height=600,im_width=600,class_num=2,aux_logits=True)
#model = AlexNet8()
#model = keras.applications.VGG19()
model = keras.applications.ResNet101()
model.build(input_shape=(1,600,600,1))
model.summary()
get_flops_params()