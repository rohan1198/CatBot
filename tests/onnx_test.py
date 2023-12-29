import onnxruntime as ort
import numpy as np
import cv2


def load_image(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (size, size))

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


def main():
    onnx_path = "/home/raspi/CatBot/models/yolov8m.onnx"
    session = ort.InferenceSession(onnx_path)

    img_path = "cat.jpeg"
    img_size = 640
    input_img = load_image(img_path, img_size)

    input_name = session.get_inputs()[0].name

    output = session.run(None, {input_name: input_img})

    print("Raw model output:\n", output)


if __name__ == "__main__":
    main()
