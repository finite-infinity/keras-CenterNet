import numpy as np
from PIL import Image, ImageEnhance, ImageOps

#调整图像颜色
#自动调节对比度
def autocontrast(image, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.autocontrast(image)
    image = np.array(image)[..., ::-1]
    return image

#均衡图像直方图（以便以后灰度图有均匀分布）
def equalize(image, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.equalize(image)
    image = np.array(image)[..., ::-1]
    return image

#暴晒图像（超过threshold的像素以prob进行翻转）
def solarize(image, prob=0.5, threshold=128.):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    image = Image.fromarray(image[..., ::-1])
    image = ImageOps.solarize(image, threshold=threshold)
    image = np.array(image)[..., ::-1]
    return image

#图片锐化增强 因子factor取1为原图
def sharpness(image, prob=0.5, min=0, max=2, factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]

# factor=0为黑白 1原图
def color(image, prob=0.5, min=0., max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def contrast(image, prob=0.5, min=0.2, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


def brightness(image, prob=0.5, min=0.8, max=1., factor=None):
    random_prob = np.random.uniform()
    if random_prob > prob:
        return image
    if factor is None:
        factor = np.random.uniform(min, max)
    image = Image.fromarray(image[..., ::-1])
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(factor=factor)
    return np.array(image)[..., ::-1]


class VisualEffect:
    """
    Struct holding parameters and applying image color transformation.

    Args
        solarize_threshold:
        color_factor: A factor for adjusting color.
        contrast_factor: A factor for adjusting contrast.
        brightness_factor: A factor for adjusting brightness.
        sharpness_factor: A factor for adjusting sharpness.
    """

    def __init__(
            self,
            color_factor=None,
            contrast_factor=None,
            brightness_factor=None,
            sharpness_factor=None,
            color_prob=0.5,
            contrast_prob=0.5,
            brightness_prob=0.5,
            sharpness_prob=0.5,
            autocontrast_prob=0.5,
            equalize_prob=0.5,
            solarize_prob=0.1,
            solarize_threshold=128.,

    ):
        self.color_factor = color_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor
        self.sharpness_factor = sharpness_factor
        self.color_prob = color_prob
        self.contrast_prob = contrast_prob
        self.brightness_prob = brightness_prob
        self.sharpness_prob = sharpness_prob
        self.autocontrast_prob = autocontrast_prob
        self.equalize_prob = equalize_prob
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold

    def __call__(self, image):
        """
        Apply a visual effect on the image.

        Args
            image: Image to adjust
        """
        random_enhance_id = np.random.randint(0, 4)
        if random_enhance_id == 0:
            image = color(image, prob=self.color_prob, factor=self.color_factor)
        elif random_enhance_id == 1:
            image = contrast(image, prob=self.contrast_prob, factor=self.contrast_factor)
        elif random_enhance_id == 2:
            image = brightness(image, prob=self.brightness_prob, factor=self.brightness_factor)
        else:
            image = sharpness(image, prob=self.sharpness_prob, factor=self.sharpness_factor)

        random_ops_id = np.random.randint(0, 3)
        if random_ops_id == 0:
            image = autocontrast(image, prob=self.autocontrast_prob)
        elif random_ops_id == 1:
            image = equalize(image, prob=self.equalize_prob)
        else:
            image = solarize(image, prob=self.solarize_prob, threshold=self.solarize_threshold)
        return image


if __name__ == '__main__':
    from generators.pascal import PascalVocGenerator
    import cv2

    train_generator = PascalVocGenerator(
        'datasets/VOC0712',
        'trainval',
        skip_difficult=True,
        anchors_path='voc_anchors_416.txt',
        batch_size=1
    )
    visual_effect = VisualEffect()
    for i in range(train_generator.size()):
        image = train_generator.load_image(i)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotations = train_generator.load_annotations(i)
        boxes = annotations['bboxes']
        for box in boxes.astype(np.int32):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        src_image = image.copy()
        image = visual_effect(image)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', np.concatenate([src_image, image], axis=1))
        cv2.waitKey(0)
