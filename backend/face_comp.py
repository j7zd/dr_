import cv2
import tensorflow as tf
import numpy as np
import facenet

MODEL_PATH = 'model/20180402-114759.pb'

def compare_images(image1, image2, model_path=MODEL_PATH, image_size=160, margin=0, gpu_memory_fraction=0.6):
    def load_and_align_data(image, image_size, margin):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (image_size, image_size))
        return resized_image

    images = [load_and_align_data(image1, image_size, margin), load_and_align_data(image2, image_size, margin)]
    images = np.stack(images)

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            # Calculate L2 distance
            dist = np.linalg.norm(emb[0] - emb[1])
            return dist
