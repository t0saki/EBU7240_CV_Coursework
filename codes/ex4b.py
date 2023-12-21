# image classification
import os
import glob
import cv2
import numpy as np
import random


# --------------------------------- WRITE YOUR CODE HERE ---------------------------------#

def load_images_from_folder(folder):
    images = []
    for filename in glob.glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    return images


airplane_images = load_images_from_folder(
    '../inputs/cifar-10-to-png/airplane')[:2201]
not_airplane_images = load_images_from_folder(
    '../inputs/cifar-10-to-png/not_airplane')[:2201]

airplane_labels = [1] * len(airplane_images)
not_airplane_labels = [0] * len(not_airplane_images)

images = airplane_images + not_airplane_images
labels = airplane_labels + not_airplane_labels

combined = list(zip(images, labels))
random.shuffle(combined)
shuffled_images, shuffled_labels = zip(*combined)

X_train = shuffled_images[:2000]
X_test = shuffled_images[2001:2201]
y_train = shuffled_labels[:2000]
y_test = shuffled_labels[2001:2201]

X_train = np.array(X_train)
X_test = np.array(X_test)

X_train = np.array([img.transpose((2, 0, 1)) for img in X_train])
X_test = np.array([img.transpose((2, 0, 1)) for img in X_test])
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
y_train = np.array(y_train)
y_test = np.array(y_test)


class CustomSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=10000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * \
                        (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


def extract_hog_features(images):
    hog_features = []
    for image in images:
        image = image.reshape(3, 32, 32).transpose(
            [1, 2, 0])  # Convert to HxWxC format
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_descriptor = cv2.HOGDescriptor(
            (32, 32), (16, 16), (8, 8), (8, 8), 9)
        hog_feature = hog_descriptor.compute(gray_image)
        hog_features.append(hog_feature)
    return np.array(hog_features).squeeze()


X_train_rgb = X_train.reshape(X_train.shape[0], -1)
X_test_rgb = X_test.reshape(X_test.shape[0], -1)

X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

svm_rgb = CustomSVM()
svm_rgb.fit(X_train_rgb, y_train)

svm_hog = CustomSVM()
svm_hog.fit(X_train_hog, y_train)

predictions_rgb = svm_rgb.predict(X_test_rgb)
predictions_hog = svm_hog.predict(X_test_hog)

print("Classification results with RGB features:")
print(predictions_rgb)

print("Classification results with HOG features:")
print(predictions_hog)


def calculate_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


predictions_rgb_binary = (predictions_rgb > 0).astype(int)
predictions_hog_binary = (predictions_hog > 0).astype(int)

TP_rgb, TN_rgb, FP_rgb, FN_rgb = calculate_confusion_matrix(
    y_test, predictions_rgb_binary)
print("Confusion matrix with RGB features:")
print(f"TP: {TP_rgb}, TN: {TN_rgb}, FP: {FP_rgb}, FN: {FN_rgb}")
print(f"Accuracy: {(TP_rgb + TN_rgb) / (TP_rgb + TN_rgb + FP_rgb + FN_rgb)}")

TP_hog, TN_hog, FP_hog, FN_hog = calculate_confusion_matrix(
    y_test, predictions_hog_binary)
print("Confusion matrix with HOG features:")
print(f"TP: {TP_hog}, TN: {TN_hog}, FP: {FP_hog}, FN: {FN_hog}")
print(f"Accuracy: {(TP_hog + TN_hog) / (TP_hog + TN_hog + FP_hog + FN_hog)}")


def display_detected_images(images, labels, preds, target_label, title, count=5):
    detected = images[labels == preds]
    for i in range(min(len(detected), count)):
        image = detected[i].reshape(3, 32, 32).transpose([1, 2, 0])
        cv2.imshow(f"{title} {i+1}", image)
        os.makedirs("../results/ex4b_results/", exist_ok=True)
        cv2.imwrite(f"../results/ex4b_results/{title} {i+1}.jpg", image)


display_detected_images(X_test, y_test, predictions_rgb,
                        "Airplane", "RGB Detected Airplanes")

display_detected_images(X_test, y_test, np.invert(
    predictions_rgb.astype(bool)), "Not Airplane", "RGB Detected Not Airplanes")

display_detected_images(X_test, y_test, predictions_hog,
                        "Airplane", "HOG Detected Airplanes")

display_detected_images(X_test, y_test, np.invert(
    predictions_hog.astype(bool)), "Not Airplane", "HOG Detected Not Airplanes")

cv2.waitKey(0)
cv2.destroyAllWindows()

##########################################################################################
