```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```

# 1. IMPORTING LIBRARIES


```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

```

# 2. Loading Data

## 2.a) Making list of all labels


```python
def get_labels(directory):
    Labels = []
    
    for labels in os.listdir(directory):
        Labels.append(labels)
        
    return Labels
    
```


```python
allLabels = get_labels('data/flowers')
```


```python
allLabels
```




    ['daisy', 'dandelion', 'flowers', 'rose', 'sunflower', 'tulip']



## 2.b) Making list of images and their labels


```python
def get_images(directory):
    Images = []
    Labels = []
    
    for labels in os.listdir(directory):
        label = allLabels.index(labels)
        
        for image_file in os.listdir(directory+labels):
            image = cv2.imread(directory+labels+r'/'+image_file)
            if image is not None:
                image = cv2.resize(image,(250,250))
                
                Images.append(image)
                Labels.append(label)
            else:
                print(directory+labels+r'/'+image_file)
            
            
    return shuffle(Images,Labels)
```


```python
Images,Labels = get_images('../input/flowers-recognition/flowers/flowers/')
```

    ERROR:root:Internal Python error in the inspect module.
    Below is the traceback from this internal error.
    
    

    Traceback (most recent call last):
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\IPython\core\interactiveshell.py", line 3326, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "<ipython-input-27-be123e00a2f4>", line 1, in <module>
        Images,Labels = get_images('../input/flowers-recognition/flowers/flowers/')
      File "<ipython-input-26-fb77b4b2d0b5>", line 5, in get_images
        for labels in os.listdir(directory):
    FileNotFoundError: [WinError 3] The system cannot find the path specified: '../input/flowers-recognition/flowers/flowers/'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\IPython\core\interactiveshell.py", line 2040, in showtraceback
        stb = value._render_traceback_()
    AttributeError: 'FileNotFoundError' object has no attribute '_render_traceback_'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\IPython\core\ultratb.py", line 1101, in get_records
        return _fixed_getinnerframes(etb, number_of_lines_of_context, tb_offset)
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\IPython\core\ultratb.py", line 319, in wrapped
        return f(*args, **kwargs)
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\IPython\core\ultratb.py", line 353, in _fixed_getinnerframes
        records = fix_frame_records_filenames(inspect.getinnerframes(etb, context))
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\inspect.py", line 1502, in getinnerframes
        frameinfo = (tb.tb_frame,) + getframeinfo(tb, context)
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\inspect.py", line 1460, in getframeinfo
        filename = getsourcefile(frame) or getfile(frame)
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\inspect.py", line 696, in getsourcefile
        if getattr(getmodule(object, filename), '__loader__', None) is not None:
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\inspect.py", line 733, in getmodule
        if ismodule(module) and hasattr(module, '__file__'):
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\tensorflow\__init__.py", line 50, in __getattr__
        module = self._load()
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\tensorflow\__init__.py", line 44, in _load
        module = _importlib.import_module(self.__name__)
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\importlib\__init__.py", line 127, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
      File "<frozen importlib._bootstrap>", line 983, in _find_and_load
      File "<frozen importlib._bootstrap>", line 953, in _find_and_load_unlocked
      File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
      File "<frozen importlib._bootstrap>", line 1006, in _gcd_import
      File "<frozen importlib._bootstrap>", line 983, in _find_and_load
      File "<frozen importlib._bootstrap>", line 967, in _find_and_load_unlocked
      File "<frozen importlib._bootstrap>", line 677, in _load_unlocked
      File "<frozen importlib._bootstrap_external>", line 728, in exec_module
      File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
      File "C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\tensorflow_core\__init__.py", line 43, in <module>
        from . _api.v2 import autodiff
    ImportError: cannot import name 'autodiff' from 'tensorflow_core._api.v2' (C:\Users\daisycharlie\Anaconda3\envs\tensor2_real\lib\site-packages\tensorflow_core\_api\v2\__init__.py)
    


    ---------------------------------------------------------------------------



```python
bi = Images
bl = Labels

```


```python
Images = bi
Labels = bl
```

converting images to numpy array


```python
Images = np.array(Images)
Labels = np.array(Labels)
```


```python
Images.shape
```


```python
Labels.shape
```

## 2.c) Image Preprocessing for training


```python
#convert images to grayscal
Images = np.sum(Images/3,axis=3,keepdims=True)
```


```python
plt.imshow(Images[0].squeeze(),cmap='gray')
```


```python
#normalize images
Images = Images/255
```


```python
Images.min()
```


```python
Images.max()
```

# 3. Splitting data

test and train split


```python
x_train,x_valid,y_train,y_valid = train_test_split(Images,Labels,test_size=0.2,random_state=42)
```

train and validation split


```python
#x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,
#                                                test_size=0.2,random_state=42)
```


```python
x_train.shape
```


```python
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_datagen.fit(x_train)
```

# 4.Creating a CNN Model


```python
#output = (input-filter +1)/stride
model = Sequential()
#1st convolutional layer
model.add(Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='relu',input_shape=(150,150,3)))#(150-5+1)/1
model.add(MaxPool2D())#output = 73
model.add(BatchNormalization())#o = 73
model.add(Dropout(0.2))#o = 73

#2nd convolutional layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(73-3+1)/1
model.add(MaxPool2D(strides=(2,2)))#output = 35
model.add(BatchNormalization())#o = 35
model.add(Dropout(0.2))#o = 35

#3rd convolutional layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(35-3+1)/1
model.add(MaxPool2D(strides=(2,2)))#output = 16
model.add(BatchNormalization())#o = 16
model.add(Dropout(0.2))#o = 16
model.summary()

#4th convolutional layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(16-3+1)/1
model.add(MaxPool2D(strides=(2,2)))#output = 16
model.add(BatchNormalization())#o = 16
model.add(Dropout(0.2))#o = 16
model.summary()

#5th convolutional layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(16-3+1)/1
model.add(MaxPool2D(strides=(2,2)))#output = 16
model.add(BatchNormalization())#o = 16
model.add(Dropout(0.2))#o = 16

#6th convolutional layer
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))#(16-3+1)/1
model.add(MaxPool2D(strides=(2,2)))#output = 16
model.add(BatchNormalization())#o = 16
model.add(Dropout(0.2))#o = 16

model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())




# Add output layer
model.add(Dense(5,activation="softmax"))
model.summary()
```

compile model


```python
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])
```


```python
#history = model.fit(x_train,y_train,epochs=50,validation_data=(x_val,y_val))
```


```python
2766/86
```


```python
#historyd = model.fit_generator(train_datagen.flow(x_train,y_train,batch_size=32),epochs=100,validation_data=(x_val,y_val))
```

1. no scaling etc output - 55-60
2. added another set of dense layer neuron - accuracy slightly increase to 62
3. added parameter padding='same' to each conv layer - not much differrence
4. increasing kernels and adding dense layers - accuracy remained roughly same , resulted in overfitting
5. chenging learning rate-accuracy reduced
6. increased image size to 256-accuracy reduced
7. added another conv layer with increased size - accuracy reduced
8. sent input data in rgb format instead of grayscale - accuracy increases
9. following operations were applied - gave better results
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
10. adding batch size and increaseing epochs to 80 - ccuracy between 65 and 75
11. adding more data to training set and increasing epochs to 100 - accuracy increased to 80%


# ----------------- with transfer learning ----------------------------

## VGG16

resetting the dataset


```python
Images = bi
Labels = bl
```


```python
Images = np.array(Images)
Labels = np.array(Labels)
```


```python
Labels
```


```python
Images = Images/255
```


```python
Images.shape
```


```python
import keras
Labels = keras.utils.to_categorical(Labels,num_classes=5,dtype='int32')
```


```python
x_train,x_test,y_train,y_test = train_test_split(Images,Labels,test_size=0.2,random_state=42)
```

splitting data


```python
x_train,x_test,y_train,y_test = train_test_split(Images,Labels,test_size=0.2,random_state=42)
```


```python
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.15)
```


```python
x_train.shape
```


```python
from keras.applications.vgg16 import VGG16
base_model = VGG16(weights='imagenet',include_top=False,input_shape=(250,250,3))
#base_model = MobileNetV2(input_shape=(150,150,3),weights='imagenet',include_top=False)
```


```python
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
```


```python
base_model
```


```python
base_model.summary()
```


```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(5,activation='softmax')(x)

model = Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
```


```python
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```


```python
model.fit(x_train,y_train,batch_size=32,epochs = 50,validation_data=(x_test,y_test))
```

1. after using base_model accuracy was reached in 10 epochs but it remained 60-65 on average
2. dataset increased - validation accuracy still not going beyond 65
3. categorical encoding done on labels - accuracy still not crossing 60
4. reduced batch_size to 32 - no change
5. trying Xception model - accuracy went till 70
6. trying vgg 16 - ACCURACY WENT TILL 80 IN 15-20 EPOCHS
7. trying vgg19 - slightly better than vgg 16
8. trying resnet50-horrible
9. trying resnet101-horrible
10. InceptionResNetV2 - accuracy did not go beyond 70
11. mobileNet - accuracy between 75-80
12. MobileNetV2 - acc = 65-70


```python

```
