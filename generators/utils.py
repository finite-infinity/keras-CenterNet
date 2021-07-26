import numpy as np
import cv2

#仿射变换
def get_affine_transform(center,
                         scale,
                         output_size,
                         rot=0.,
                         inv=False):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list) and not isinstance(scale, tuple):
        scale = np.array([scale, scale], dtype=np.float32)

    if not isinstance(output_size, np.ndarray) and not isinstance(output_size, list) and not isinstance(output_size,
                                                                                                        tuple):
        output_size = np.array([output_size, output_size], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    src_h = scale_tmp[1]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_h * -0.5], rot_rad)
    dst_dir = np.array([0, dst_h * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

#画高斯半径
#在设置GT box的heat map的时候，我们不能仅仅只在top-left/bottom-right的位置设置标签，因为除了GT框，绿色的框其实也能很好的包围目标。
#所以如果在检测中得到想绿色的这样的框的话，我们也给它保留下来。
#只要预测的corners在top-left/bottom-right点的某一个半径r内，并且其与GTbox的IOU大于一个阈值(一般设为0.7)，我们将将这些点的标签不直接置为0
#那置为多少呢？可以通过一个温和的方式来慢慢过渡，所以采用二维的高斯核未尝不可。
#总之最后会生成一个个越往外标签越小的点，作为中心点的候选区域

#定义辐射区域（即保留一定标签的区域）
#椭圆
def draw_gaussian(heatmap, center, radius_h, radius_w, k=1):
    diameter_h = 2 * radius_h + 1
    diameter_w = 2 * radius_w + 1  #辐射椭圆直径
    gaussian = gaussian2D((diameter_h, diameter_w), sigma_w=diameter_w / 6, sigma_h=diameter_h / 6)   # sigma是一个与直径相关的参数
    # 一个圆对应内切正方形的高斯分布

    x, y = int(center[0]), int(center[1])   #中心点坐标

    height, width = heatmap.shape[0:2]
    
    #上下左右距中心距离（防止越界）
    left, right = min(x, radius_w), min(width - x, radius_w + 1)   #左端距x距离
    top, bottom = min(y, radius_h), min(height - y, radius_h + 1)  #辐射框底端距y距离（不跑出图外）  顶端距y距离（不高于GT框）

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]  #保留图的mask
    # 取[abs(radius_h - y):max(y, radius_h)+min(height - y, radius_h + 1), abs(radius_w - x):max(x, radius_w)+min(width - x, radius_w + 1)
    # 将高斯分布结果约束在边界内
    masked_gaussian = gaussian[radius_h - top:radius_h + bottom, radius_w - left:radius_w + right] 
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，相当于不断的在heatmap基础上添加关键点的高斯，
        # 即同一种类型的框会在一个heatmap某一个类别通道上面上面不断添加。
        # 最终通过函数总体的for循环，相当于不断将目标画到heatmap
    return heatmap


#长宽半径相同的圆
def draw_gaussian_2(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D_2((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def gaussian2D(shape, sigma_w=1, sigma_h=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-((x * x) / (2 * sigma_w * sigma_w) + (y * y) / (2 * sigma_h * sigma_h)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian2D_2(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    det_h, det_w = det_size
    rh = 0.1155 * det_h
    rw = 0.1155 * det_w
    return rh, rw


 #三种情况的r（大、小、错位）
def gaussian_radius_2(det_size, min_overlap=0.7):
    height, width = det_size
    #错位 （b正负无所谓）
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / (2 * a1)
    
    #小
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / (2 * a2)
    
    #大
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)  #返回最小的辐射半径


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


if __name__ == '__main__':
    gaussian2D((3, 3))
