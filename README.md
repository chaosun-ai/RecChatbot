# Example recbot

This is an prototype of using chatbot to do cloth recommendation.

## Get started

1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. Put articles.csv in the project folder. (I didn't put the file on git.)
5. Create an .env file. In that file, add the following info.
    OPENAI_API_KEY=**
    OPENAI_API_TYPE=**
    OPENAI_API_VERSION=**
    OPENAI_API_BASE=**
    EMBEDDING_DEPLOYMENT_NAME =**
6. Run embedding.ipynb to generate articles_embedding.csv.
7. In the terminal, run 'streamlit run recbot.py'.
8. A page will be opened in your browser. If not, open http://localhost:8501/ in your browser.



Testing case:

Question: Could you please suggest my daughter a birthday cloth.

Following question: Her age is 16 and she really like modern fashion.