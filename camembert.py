import sys

from tqdm.auto import tqdm
from datasets import Dataset
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from pandas import DataFrame
from transformers import TextClassificationPipeline, DataCollatorWithPadding
from transformers import TFCamembertForSequenceClassification, CamembertTokenizer
from transformers.pipelines.base import KeyDataset

from main import get_dataframe


def prepare_dataset(document: DataFrame) -> DataFrame:
    document.drop(columns=['movie', 'name', 'user_id'], inplace=True)
    document['commentaire'].fillna(" ", inplace=True)

    if 'note' in document.columns:
        document['note'].fillna("0,5", inplace=True)
        document['note'].apply(lambda note: int((float(note.replace(',', '.')) - 0.5)/2))

    document.rename(columns={"commentaire": "text", "note": "label"}, inplace=True)

    return document


def preprocess_function(ds: Dataset):
    return tokenizer(ds['text'], truncation=True)


def test(document_dataframe: DataFrame, tokenizer: CamembertTokenizer):
    model = TFCamembertForSequenceClassification.from_pretrained('models/camembert-finetuned/')
    batch_size = 1
    num_epochs = 3
    batches_per_epoch = len(document_dataframe)
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=2e-5
    )

    model.compile(optimizer=optimizer)

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    document_ds = Dataset.from_pandas(document_dataframe)
    predicted_class = []
    for result in tqdm(pipe(KeyDataset(document_ds, "text"), truncation=True), total=len(document_dataframe)):
        predicted_class.append(result['label'])

    render = DataFrame({'review_id': document_dataframe['review_id'], 'note': predicted_class})
    render.to_csv("data/render.txt", sep=" ", header=False, index=False)


def train(document_dataframe: DataFrame, tokenizer: CamembertTokenizer):

    # Attention les deux lignes ci-dessous seront a réévaluer
    id2label = {0: "0,5", 1: "1,0", 2: "1,5", 3: "2,0", 4: "2,5", 5: "3,0", 6: "3,5", 7: "4,0", 8: "4,5", 9: "5,0"}
    label2id = {"0,5": 0, "1,0": 1, "1,5": 2, "2,0": 3, "2,5": 4, "3,0": 5, "3,5": 6, "4,0": 7, "4,5": 8, "5,0": 9}

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    document_ds = Dataset.from_pandas(document_dataframe)
    document_ds = document_ds.class_encode_column("label")
    tokenized_document = document_ds.map(preprocess_function, batched=True)

    tokenized_document = tokenized_document.train_test_split(test_size=0.2, stratify_by_column="label")

    batch_size = 16
    num_epochs = 3
    batches_per_epoch = len(tokenized_document['train'])
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=2e-5
    )

    model = TFCamembertForSequenceClassification.from_pretrained(
        "camembert-base", num_labels=10, id2label=id2label, label2id=label2id
    )

    tf_train_set = model.prepare_tf_dataset(
        tokenized_document['train'],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_document["test"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)  # No loss argument!

    checkpoint = ModelCheckpoint(
        filepath="./models/camembert-allocine/",
        monitor="accuracy",
        mode='max',
        save_best_only=True
    )

    model.fit(tf_train_set, validation_data=tf_validation_set, epochs=num_epochs, callbacks=[checkpoint])
    model.save("./models/camembert-fine-tuned")
    model.save_weights("./models/camembert-fine-tuned-weights")


if __name__ == '__main__':

    if len(sys.argv) == 2:
        print("Testing model...")
        path = 'data/test'
    else:
        print("Training model...")
        path = 'data/dev'

    document = get_dataframe(path)
    document = prepare_dataset(document)

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    if len(sys.argv) == 2:
        tf.config.set_visible_devices([], 'GPU')
        test(document, tokenizer)
    else:
        train(document, tokenizer)




