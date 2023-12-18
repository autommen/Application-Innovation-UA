from transformers import CamembertModel, CamembertTokenizer, pipeline

from main import get_dataframe

if __name__ == '__main__':
    path = 'data/test'
    document = get_dataframe(path)
    tokenizer = CamembertTokenizer.from_pretrained("./models/camembert-base")

    """
    for index, row in document.head(1000).iterrows():
        masked_line = 'Commentaire : '+row['commentaire'][:1500]+' Note : <mask> /5'
        print(masked_line)
        result = camembert_fill_mask(masked_line)
        print(result[0]['token_str'])
    """