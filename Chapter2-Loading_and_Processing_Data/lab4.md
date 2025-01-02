# Chapter 2: LAB - 4: Loading Text Data for TensorFlow Models

In this exercise, you'll learn how to load in image data for batch processing. The dataset, ‘drugLibTrain_raw.tsv’, contains information related to patient reviews on specific drugs along with related conditions. Reviews and ratings are grouped into reports on the three aspects benefits, side effects and overall comment. n this exercise, you will load in text data for batch processing. You will apply a pretrained model from TensorFlow Hub to perform a word embedding on the patient reviews. You are required to work on the commentsReview field only as that contains text data.

#### Task 1 - Open Google Collab

1. Open Google Colab:

- Go to Google Colab.
- If prompted, sign in with your Google account.

2. Create a New Notebook:
- Once you're logged in, click on File in the top left corner.
- Select New Notebook.

#### Task 2 - Import TensorFlow

Import the TensorFlow library to access its functionalities, such as creating datasets and processing text data.

```python
import tensorflow as tf
```

#### Task 3: Load a Tab-Delimited Dataset into a TensorFlow Dataset Object

- Use TensorFlow's make_csv_dataset function to load a tab-delimited file (drugLibTrain_raw.tsv) into a dataset object.
- Set the batch_size to 1 for initial data processing, and specify the field_delim argument as '\t' to handle the tab-delimited format.

```python
df = tf.data.experimental.make_csv_dataset(
    r'/content/sample_data/dataset/drugLibTrain_raw.tsv',
    batch_size=1, 
    field_delim='\t'
)


```

#### Task 4: Define a Function to Shuffle, Repeat, and Batch the Dataset.

Create a function prep_ds to preprocess the dataset by:
Shuffling: Randomizing the order of data for training.
Repeating: Ensuring the dataset continues indefinitely for training epochs.
Batching: Combining multiple examples into batches for efficient processing.


```python
def prep_ds(ds, shuffle_buffer_size=1024, batch_size=32):
    # Shuffle the dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat the dataset
    ds = ds.repeat()
    # Batch the dataset
    ds = ds.batch(batch_size)
    return ds

```

#### Task 5: Preprocess the Dataset


Apply the prep_ds function to the dataset object to shuffle, repeat, and batch the data. Set batch_size=5 to process 5 examples in each batch.

```python
ds = prep_ds(df, batch_size=5)
```

#### Task 6:  Print the First Batch

Retrieve and print the first batch of the dataset to verify the data. The output will display the data in tensor format, representing both the features and labels.
python
Copy code


```python
for x in ds.take(1):
    print(x)
```

#### Task 7: Import a Pretrained Word Embedding Model from TensorFlow Hub

Load a pretrained word embedding model from TensorFlow Hub and create a Keras layer. This model will be used to convert text data into numerical embeddings for machine learning.

```python
import tensorflow_hub as hub

# Load a pretrained embedding model
hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2", input_shape=[], dtype=tf.string, trainable=False)


```

#### Task 8: Apply the Pretrained Embedding Model to the Text Data

Extract a batch of data from the dataset and flatten the tensor corresponding to the commentsReview field.
Apply the pretrained embedding model (hub_layer) to convert text into numerical embeddings.
Print the embeddings for the first batch.


```python
for x in ds.take(1):
    print(hub_layer(tf.reshape(x['commentsReview'], [-1])))

```
The preceding output represents the embedding vectors for the first batch of drug reviews. The specific values may not mean much at first glance but encoded within the embeddings is contextual information based on the dataset that the embedding model was trained upon. The batch size is equal to 5 and the embedding vector size is 20, which means the resulting size, after applying the pretrained layer, is 5x20.
In this exercise, you learned how to import tabular data that might contain a variety of data types. You took the commentsReview field and applied a pretrained word embedding model to convert the text into a numerical tensor. Ultimately, you pre-processed and batched the text data so that it was appropriate for large-scale training. This is one way to represent text so that it can be input into machine learning models in TensorFlow. In fact, other pretrained word embedding models can be used and are available on TensorFlow Hub. You will learn more about how to utilize TensorFlow Hub in the next chapter.


