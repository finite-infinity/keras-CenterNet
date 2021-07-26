import tensorflow as tf
import keras.backend as K
from keras.losses import mean_absolute_error

#heatmap_loss 
def focal_loss(hm_pred, hm_true):
    pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)  #正样本的mask 将true false转换成float（0/1）
    neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)  #改进focal_loss添加的项，抑制高斯点附近的负样本

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, 2) * pos_mask  #只剩下正样本的loss
    neg_loss = -tf.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)
    
    #cond(exp, lambda:, lambda:)达到if... else效果 
    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss) #若num_pos>0, cls_loss=(pos_loss + neg_loss) / num_pos
    return cls_loss

# 正则l1 loss
# indices = ys * w + xs
def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)     # wh：shape=(max_objects, 2)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))  # 将mask第3维复制为原来两倍（为了乘wh）
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

# arg:[y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input]
# shape:
#     image_input = Input(shape=(None, None, 3))  
#     hm_input = Input(shape=(output_size, output_size, num_classes))  
#     wh_input = Input(shape=(max_objects, 2))
#     reg_input = Input(shape=(max_objects, 2))
#     reg_mask_input = Input(shape=(max_objects,))
#     index_input = Input(shape=(max_objects,))
def loss(args):
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices = args
    hm_loss = focal_loss(hm_pred, hm_true)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    return total_loss
