from keras_resnet.models import ResNet18, ResNet34, ResNet50
from keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.layers import UpSampling2D, Concatenate
from keras.models import Model
from keras.initializers import normal, constant, zeros
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf

from losses import loss

#将多个识别框变为一个（非极大值抑制，抑制置信度低的box）
def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')  #置信度最大的box
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat)) #hmax和heat相同的元素不变，其他的替换成0
    return heat


#每一张图像，每一个类别，在w*h维度上，取前K=100个
def topk(hm, max_objects=100):
    hm = nms(hm)  #先过滤重复度高的box
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]  #batch
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)                                                                          
    scores, indices = tf.nn.top_k(hm, k=max_objects)   #找到最后一维最大的100个数（即score）  
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)                                   index=c*[w*h + w*y + x] + class
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c   #class
    xs = indices // c % w     #indices // c代表box号（除个）  %w代表score的x坐标
    ys = indices // c // w    #score的y坐标
    indices = ys * w + xs     # indices//c - indices // c % w + indices // c % w = indices//c
    return scores, indices, class_ids, xs, ys   #x,y为box的位置


#挑出每个class中score>score_threshold，iou<iou_threshold的元素（个数也有限制），如果batch_size>100，用0填充被挑出去的元素，保持batch_size
def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=20, max_objects=100,
                        iou_threshold=0.5, score_threshold=0.1):
    
    #挑出每个class中score>score_threshold，iou<iou_threshold的元素
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold)) #保留score>阈值（0.1）的batch_item_detections
    detections_per_class = []
    for cls_id in range(num_classes):
        # (num_keep_this_class_boxes, 4) score 大于 score_threshold 的当前 class 的 boxes
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id)) #保留属于cls_id类的元素
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)  #按照参数scores（data[:4]）的降序贪婪的选择边界框的子集，返回元素索引
                                                                                      #最大输出元素个数为max_objects_per_class
        class_detections = K.gather(class_detections, nms_keep_indices)  #找到索引是nms_keep_indices的元素
        detections_per_class.append(class_detections)

    batch_item_detections = K.concatenate(detections_per_class, axis=0)  #按batch_size拼接数据

    def filter():
        #挑出max_objects个score最高的元素
        nonlocal batch_item_detections  #声明变量是外部嵌套函数的变量
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        #挑完后元素个数变少，用0填充
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)  #按照第一个维度（batch）在最后一个元素后添加batch_item_num_pad个0
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections

#利用hm（heatmap）得到的topk inds（维度num_class*100），将 对应位置wh，reg值提取出来；再利用xs和ys计算bbox：
#[topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids]
def decode(hm, wh, reg, max_objects=100, nms=True, num_classes=20, score_threshold=0.1):
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)    #从heatmap中取出score最高的100个点
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))  #残差，reshape成（batch, h*w, channel=2） 即每张图的残差
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))     #宽高 reshape成（batch, h*w, channel=2） 每张图的宽高
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1) 
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)  #取出wh中对应indices下标的数据
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]  
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]     #把中心点残差加上，减小误差
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2    #得到box：（xmin,ymin,xmax,ymax）
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:     #是否nms处理
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections


def centernet(num_classes, backbone='resnet50', input_size=512, max_objects=100, score_threshold=0.1, nms=True):
    assert backbone in ['resnet18', 'resnet34', 'resnet50']
    output_size = input_size // 4   #4倍下采样
    image_input = Input(shape=(None, None, 3))  
    hm_input = Input(shape=(output_size, output_size, num_classes))  
    wh_input = Input(shape=(max_objects, 2))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))   #构建输入层

    if backbone == 'resnet18':
        resnet = ResNet18(image_input, include_top=False, freeze_bn=True)
    elif backbone == 'resnet34':
        resnet = ResNet34(image_input, include_top=False, freeze_bn=True)
    else:
        resnet = ResNet50(image_input, include_top=False, freeze_bn=True)

    # C5 (b, 16, 16, 512)
    C2, C3, C4, C5 = resnet.outputs   #resnet的输出
    
    #dropout
    C5 = Dropout(rate=0.5)(C5)
    C4 = Dropout(rate=0.4)(C4)
    C3 = Dropout(rate=0.3)(C3)
    C2 = Dropout(rate=0.2)(C2)
    x = C5    #Dropout=0.5

    # decoder 计算hm wh reg
    x = Conv2D(256, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([C4, x])
    x = Conv2D(256, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    # (b, 32, 32, 512)
    x = ReLU()(x)

    x = Conv2D(128, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([C3, x])
    x = Conv2D(128, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    # (b, 64, 64, 128)
    x = ReLU()(x)

    x = Conv2D(64, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(UpSampling2D()(x))
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([C2, x])
    x = Conv2D(64, 3, padding='same', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    # (b, 128, 128, 512)
    x = ReLU()(x)

    # hm header   batch * numclass * 128 * 128
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y1 = BatchNormalization()(y1)
    y1 = ReLU()(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y1)

    # wh header  batch * 2 * 128 * 128
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y2 = BatchNormalization()(y2)
    y2 = ReLU()(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y2)

    # reg header  batch * 2 * 128 * 128
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
    y3 = BatchNormalization()(y3)
    y3 = ReLU()(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(y3)

    loss_ = Lambda(loss, name='centernet_loss')(
        [y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input]) #搭建Lambda层 计算loss ，继承tf.keras.layers.Layer并重写config更稳妥
    #input为lable（会从model输入）
    
    # input_layer = keras.Input(shape), conv = conv_layer(input_layer), ..., output_layer = conv_layer(conv)
    # model = Model(input=input_layer, output=output_layer)
    # 搭建训练用model
    # model用于生成loss
    model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # detections = decode(y1, y2, y3) hm wh reg
    # detection:[topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids]
    detections = Lambda(lambda x: decode(*x,
                                         max_objects=max_objects,
                                         score_threshold=score_threshold,
                                         nms=nms,
                                         num_classes=num_classes))([y1, y2, y3])
    #从resnet输入img_input开始一路搭建的网络（detec时会加载权重）
    prediction_model = Model(inputs=image_input, outputs=detections)  # 测试（推断）用model,最终输出为推断值[topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids]
    debug_model = Model(inputs=image_input, outputs=[y1, y2, y3])     # 输入图像到输出 hm wh reg 再输入model计算loss
    return model, prediction_model, debug_model
