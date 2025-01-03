# Chapter 2: LAB - 5: Loading Audio Data for TensorFlow Models.

In this exercise, you'll learn how to load in audio data for batch processing. The dataset, data_speech_commands_v0.02, contains speech samples of people speaking the word zero for exactly 1 second with a sample rate of 44.1 kHz, meaning that for every second, there are 44,100 data points. You will apply some common audio pre-processing techniques, including converting the data into the Fourier domain, sampling the data to ensure the data has the same size as the model, and generating MFCCs for each audio sample. This will generate a pre-processed dataset object that can be input into a TensorFlow model for training.

#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Import TensorFlow

Import TensorFlow and OS to handle audio processing and file operations.

```python
import tensorflow as tf
import os

```

#### Task 3: Create a Function to Load Audio

Define a function to load and preprocess audio files into TensorFlow tensors.

```python
def load_audio(file_path, sample_rate=44100):
    audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio, desired_channels=-1, desired_samples=sample_rate)
    return tf.transpose(audio)


```

#### Task 4: Load Dataset Paths

 Use os.listdir to create a list of file paths for the audio dataset.


```python
prefix = "dataset/data_speech_commands_v0.02/zero/"
paths = [os.path.join(prefix, path) for path in os.listdir(prefix)]

```

#### Task 5: Visualize an Audio Sample

Load an audio file and plot its waveform to inspect its signal.

```python
import matplotlib.pyplot as plt
audio = load_audio(paths[0])
plt.plot(audio.numpy().T)
plt.xlabel('Sample')
plt.ylabel('Value')

```

#### Task 6:  Create MFCC Generator Function

Write a function to convert raw audio into Mel-frequency cepstral coefficients (MFCCs).


```python
def apply_mfccs(audio, sample_rate=44100, num_mfccs=13):
    stfts = tf.signal.stft(audio, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        80, stfts.shape[-1], sample_rate, 80.0, 7600.0)
    mel_spectrograms = tf.tensordot(spectrograms, mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfccs]

```

#### Task 7: Visualize MFCCs

Apply the MFCC function to a sample and visualize the output.

```python
mfcc = apply_mfccs(audio)
plt.pcolor(mfcc.numpy()[0])
plt.xlabel('MFCC log coefficient')
plt.ylabel('Sample Value')



```

#### Task 8: Enable Efficient Processing

Load TensorFlow’s AUTOTUNE for faster data pipeline optimization.


```python
AUTOTUNE = tf.data.experimental.AUTOTUNE

```

#### Task 9: Create a Dataset Preparation Function

Write a function to shuffle, preprocess, batch, and prefetch the dataset.

```python
def prep_ds(ds, shuffle_buffer_size=1024, batch_size=64):
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.map(load_audio, num_parallel_calls=AUTOTUNE)
    ds = ds.map(apply_mfccs)
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

```

#### Task 10: Prepare Training Dataset

Create a TensorFlow dataset object and apply the preparation function.

```python
ds = tf.data.Dataset.from_tensor_slices(paths)
train_ds = prep_ds(ds)

```

#### Task 11: Prepare Training Dataset
Print and inspect the structure of the first batch of preprocessed data.

```python
for x in train_ds.take(1):
    print(x)

```
In this exercise, you imported audio data. You processed the dataset and batched the dataset so that it is appropriate for large-scale training. This method was a comprehensive approach in which the data was loaded and converted into the frequency domain, spectrograms were generated, and then finally the MFCCs were generated.
