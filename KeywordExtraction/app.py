from flask import Flask, request, jsonify



from flask_restful import Resource, Api, reqparse
from urllib.request import urlopen
import bs4
import validators
# import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer

n_gram_range = (1, 1)
stop_words = "english"
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L12-v2')
from sklearn.metrics.pairwise import cosine_similarity

top_n = 8

def keywoext(text):
    # print(text)
    doc=text
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names()
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    return dict(map(lambda x: (x,'False'), keywords))

app = Flask(__name__)
# api = Api(app)
@app.route('/')
def index():
    return "<h1>Topic Modelling Up !!</h1>"
    
@app.route('/', methods=['POST'])

def respond():    

# class HelloWorld(Resource):
#     def getKey(self,doc):

    # Extract candidate words/phrases
    urlin = request.args.get("linkortext", None)
    # print("goturl",urlin)
    # print("tes")
    if validators.url(urlin):
        # print("in if")
        html = urlopen(urlin).read()
        soup = BeautifulSoup(html, features="html.parser")

        # kill all script and style elements
        for script in soup(["script", "style"]):
            script.extract()    # rip it out

        # get text
        text = soup.get_text()

        # break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # print(text)
        return keywoext(text)
    
        
    else:
        # print("in else")
        return keywoext(urlin)
    

# api.add_resource(HelloWorld, '/')
if __name__ == '__main__':
    app.run(debug=True)