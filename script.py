import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import cv2
from scipy.stats import kurtosis, skew,entropy
import numpy as np
from scipy import ndimage
import statistics
import base64

data = pd.read_csv('banknote_authentication.txt', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
print(data.head())

sns.pairplot(data, hue='auth')
sns.countplot(x=data['auth'])
target_count = data.auth.value_counts()

nb_to_delete = target_count[0] - target_count[1]
data = data.sample(frac=1, random_state=42).sort_values(by='auth')
data = data[nb_to_delete:]

x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

scalar = StandardScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)

clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf.fit(x_train, y_train.values.ravel())

y_pred = np.array(clf.predict(x_test))
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                        columns=["Pred.Negative", "Pred.Positive"],
                        index=['Act.Negative', "Act.Positive"])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = round((tn+tp)/(tn+fp+fn+tp), 4)

print(f'\n Accuracy = {round(100*accuracy, 2)}%')


img = '1.jpg'

##import base64
##with open(img, "rb") as img_file:
##    my_string = base64.b64encode(img_file.read())
##print(my_string)
##
##image_code = base64.b64decode(my_string)
##with open('name.jpg', 'wb') as f:
##   f.write(image_code)

gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
norm_image = cv2.normalize(gray_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

img_blur = cv2.GaussianBlur(norm_image, (3,3), 0)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
sobelxy = cv2.imshow('sobelxy', sobelxy)

var = np.var(norm_image,axis=None)
sk = skew(gray_img, axis=None)
kur = kurtosis(gray_img, axis=None)
ent = entropy(gray_img, axis=None)
ent = ent/100



from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
result = clf.predict(np.array([[-0.91318,-2.0113,-0.19565,0.066365]]))
result = clf.predict(np.array([[var,sk,kur,ent]]))
print(result)
        
        



