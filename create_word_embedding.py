# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 19:40:27 2024

@author: pikam
"""

import pandas
import os
from pandas import DataFrame
import string
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import tqdm
import io



#une partie du code est reprise depuis : https://www.tensorflow.org/text/tutorials/word2vec


def get_dataframe(filepath: str) -> DataFrame:
    if not os.path.exists(filepath + '.csv'):
        dataframe = pandas.read_xml(filepath + '.xml')

        dataframe['commentaire'].fillna(" ", inplace=True)
        dataframe['commentaire'] = dataframe['commentaire'].str.replace(f"[{string.punctuation}]", '')
        dataframe['commentaire'].apply(lambda comment: comment.lower())
        #dataframe['commentaire'] = dataframe['commentaire'].apply(remove_stopwords)

        if 'note' in dataframe.columns:
            dataframe['note'].fillna("0,5", inplace=True)
            dataframe['note'].apply(lambda note: float(note.replace(',', '.')))

        # Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
        dataframe.to_csv(filepath + '.csv', index=False)
    else:
        dataframe = pandas.read_csv(filepath + '.csv')
    return dataframe


# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels




class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(dict_length,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(dict_length,
                                       embedding_dim,
                                       input_length=5)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots








path = '/content/drive/MyDrive/appli_innov/train'
#path = 'train_all_fr'
document = get_dataframe(path)




mots_choisis = '/content/drive/MyDrive/appli_innov/mots_choisis.txt'
dict_mots_choisis = {}
dict_mots_choisis['<pad>'] = 0  # add a padding token

with open(mots_choisis, 'r', encoding='utf-8') as mots:
    count = 1
    for line in mots :
        #if not re.search('[a-zA-Z]', line):
            #print(line.strip())
        if len(line.strip()) != 0:
            dict_mots_choisis[line.strip()] = count
            count = count + 1

print(dict_mots_choisis)
inverse_dict = {index: token for token, index in dict_mots_choisis.items()}
print(inverse_dict)
dict_length = len(dict_mots_choisis)

    
max_length = 0
sentences = []
count = 0
for index, row in document.iterrows():
    if count == 5000:
        break
    commentaire = str(row['commentaire'])
    #commentaire = remove_punctuation(commentaire)
    commentaire = commentaire.split()
    #print(commentaire)
    sentence = []
    for word in commentaire :
        if word in dict_mots_choisis :
            sentence.append(dict_mots_choisis[word])
    sentences.append(sentence)
    if len(sentence) > max_length:
        max_length = len(sentence)
    count = count +1 
print(sentences)
print(len(sentences))
print(max_length)

for i in range(len(sentences)):
    sentences[i] += [0] * (max_length - len(sentences[i]))


targets, contexts, labels = generate_training_data(
    sequences=sentences,
    window_size=2,
    num_ns=4,
    vocab_size=dict_length,
    seed=42)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

dataset = dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

embedding_dim = 100
word2vec = Word2Vec(dict_length, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])


word2vec.fit(dataset, epochs=20)

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
#vocab = vectorize_layer.get_vocabulary()
vocab = list(dict_mots_choisis.keys())

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
