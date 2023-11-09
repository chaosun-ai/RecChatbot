from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text)

    return embeddings

def calculate_cos_similarity(input_embedding, row):
    return cosine_similarity([input_embedding], [row])[0][0]

def sort_df_similarity(df, input_embedding, col_name):
    df['cos_similarity'] = df[col_name].apply(calculate_cos_similarity, args=(input_embedding,))
    df = df.sort_values(by='cos_similarity', ascending=False)
    return df
    
