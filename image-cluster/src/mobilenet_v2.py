"""
MobileNet V2 클래스 입니다. Tensorflow Hub에서 다운받은 Pre-train 모델을 사용합니다.
"""
import tensorflow as tf
import tensorflow_hub as hub


def get_encoded_image(image_path):
    encoded_image = tf.gfile.FastGFile(image_path, 'rb').read()
    return encoded_image


class MobileNet2:
    def __init__(self):
        self.module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/2"
        self.filename = tf.placeholder(tf.string, shape=[None], name='filename')
        self.encoded_images = tf.placeholder(tf.string, shape=[None], name='encoded_images')
        self.features = self.build_model()
        self.output_size = 1792

    def build_model(self):
        # build mobilenet v2 model using tensorflow hub
        image_module = hub.Module(self.module_url)
        image_size = hub.get_expected_image_size(image_module)

        def _decode_and_resize_image(encoded: tf.Tensor) -> tf.Tensor:
            decoded = tf.image.decode_jpeg(encoded, channels=3)
            decoded = tf.image.convert_image_dtype(decoded, tf.float32)
            return tf.image.resize_images(decoded, image_size)

        decoded_images = tf.map_fn(_decode_and_resize_image, self.encoded_images, tf.float32)  # type: tf.Tensor
        return image_module(decoded_images)
