# Chapter 8: LAB - 1: Classifying pizza and steak with Transfer Learning.

In this exercise, you will use transfer learning to correctly classify images as either pizzas or steak. You will use a pre-trained model, NASNet-Mobile, that is already available in TensorFlow. This model comes with pre-trained weights on ImageNet. 
Note
The original dataset used in this exercise has been provided by Google. It contains 2000 images of pizzas and steaks. It can be found here: 
https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

#### Task 1: Import the libraries.

This imports library as follows

```python
import tensorflow as tf
```

#### Task 2: Download and Extract Dataset.

 This code downloads and extracts a zip file containing the "pizza" and "steak" image dataset from a URL using TensorFlow’s get_file function. 

```python
file_url = 'https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip'
zip_dir = tf.keras.utils.get_file('pizza_and_streak.zip', origin=file_url, extract=True)

```

#### Task 3:Define Dataset Paths.  

This defines the file paths for the training and validation directories within the extracted dataset.


```python
path = pathlib.Path(zip_dir).parent / 'pizza_steak'
train_dir = path / 'train'
validation_dir = path / 'test'

```

#### Task 4: Check Dataset Sizes. 

This calculates the total number of images in the training and validation directories by counting the files in each class subdirectory (pizza and steak).

```python
total_train = len(os.listdir(train_pizza_dir)) + len(os.listdir(train_steak_dir))
total_val = len(os.listdir(validation_pizza_dir)) + len(os.listdir(validation_steak_dir))


```

#### Task 5: Initialize Image Generators.

These ImageDataGenerator instances are used to rescale images to the range [0, 1] by dividing pixel values by 255, which is common for neural network input.


```python
train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)

```

#### Task 6: Load Training and Validation Data.

This loads the images from the directories into a format suitable for training. It resizes the images to 224x224 pixels, shuffles them, and sets the class mode to "binary" since there are two classes (pizza and steak).

```python
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(img_height, img_width), class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(img_height, img_width), class_mode='binary')


```

#### Task 7: Set Random Seed for Reproducibility.

This sets the random seed for NumPy and TensorFlow to ensure reproducible results across different runs of the code.

```python
np.random.seed(8)
tf.random.set_seed(8)


```

#### Task 8: Load Pre-trained NASNet Mobile Model.

This loads the NASNet Mobile model without the top (fully connected) layers and with pre-trained weights from ImageNet, which can be used for transfer learning.

```python
base_model = NASNetMobile(include_top=False, input_shape=(img_height, img_width, 3), weights='imagenet')


```

#### Task 9: Freeze Base Model and Build New Model.

This freezes the pre-trained layers of the NASNet Mobile model (so they won’t be updated during training) and adds new layers for binary classification (a dense layer and a sigmoid activation).
 
 ```python
base_model.trainable = False
model = tf.keras.Sequential([base_model, layers.Flatten(), layers.Dense(500, activation='relu'), layers.Dense(1, activation='sigmoid')])
```

#### Task 10: Compile the Model.

This compiles the model, specifying the loss function (binary_crossentropy), optimizer (Adam with a learning rate of 0.001), and the evaluation metric (accuracy).

```python
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])

```

#### Task 11: Train the Model.

This trains the model for 5 epochs, using the training and validation data generators. It defines the steps per epoch and validation steps based on the size of the datasets.



```python
model.fit(train_data_gen, steps_per_epoch=total_train // batch_size, epochs=5, validation_data=val_data_gen, validation_steps=total_val // batch_size)

```
You can observe that the model achieved an accuracy score of 0.99 on the training set and 0.98 on the validation set. This is quite a remarkable result given that you only trained the last two layers, and it took less than a minute. This is the benefit of applying transfer learning and using pre-trained state-of-the-art models.
