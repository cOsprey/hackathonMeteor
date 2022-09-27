from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
from urllib.request import urlopen
import urllib.request
import os
import requests
import sys
import json
import bs4
import validators
from bs4 import BeautifulSoup
sys.path.append('/')
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



classes=['Apple___apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust', 'Apple___healthy','Blueberry___healthy', 'Cherry___Powdery_mildew','Cherry___healthy','Corn___Cercospora_leaf_spot Gray_leaf_spot','Corn___Common_rust_','Corn___Northern_Leaf_Blight','Corn___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Powdery_Mildew','Grape___Downy_Mildew','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight', 'Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy','None']
# plantsclass=['Apple___','BlueBerry___','Cherry_','Corn_maize___','Grape___','Peach___','Tomato___','Pepper,_bell___','Potato__', 'Squash__','Raspberry___','Strawberry__','Soyabean__','']
import tensorflow as tf
import numpy as np
from PIL import Image

class TensorflowLiteClassificationModel:
    def __init__(self, model_path, labels, image_size=299):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()
        self.labels = labels
        self.image_size=image_size

    def run_from_filepath(self, image_path):
        input_data_type = self._input_details[0]["dtype"]
        # print(input_data_type)
        image = np.array(Image.open(image_path).resize((self.image_size, self.image_size)), dtype=input_data_type)
        image = np.expand_dims(image, axis=0)
        # print(image.shape)
        if input_data_type == np.float32:
            image = image / 255.
            print("here")

        if image.shape == (1, 299, 299):
            image = np.stack(image*3, axis=0)
            print("hererere")

        return self.run(image)

    def run(self, image):
        """
        args:
          image: a (1, image_size, image_size, 3) np.array

        Returns list of [Label, Probability], of type List<str, float>
        """

        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        probabilities = np.array(tflite_interpreter_output[0])*100
        p=probabilities
        # print(probabilities)
        # print(labels)

        # create list of ["label", probability], ordered descending probability
        label_to_probabilities = []
        for i, probability in enumerate(probabilities):
            label_to_probabilities.append([self.labels[i], float("{:.8f}".format(probability))])
        return sorted(label_to_probabilities, key=lambda element: element[1])

            
model = TensorflowLiteClassificationModel("Model1.tflite",classes)
app = Flask(__name__)
# api = Api(app)

@app.route('/')
def index():
    return "<h1>Plant Disease Prediction !!</h1>"
    
@app.route('/', methods=['POST'])

def respond():

    json_data = request.get_json(force=True)
    cropname = json_data['crop']
    userLocation = json_data['userLocation']
    imgurls= json_data['images']

    res=[]
    #args = parser.parse_args()
    for i in range(0,len(imgurls)):
        print(i,imgurls[i])
        fp=str(i)+".jpg"
        print(fp)
        try:
            url = imgurls[i]
            r = requests.get(url)
            print("errir")
            print(r.status_code)
            with open(fp, 'wb') as outfile:
                outfile.write(r.content)
            # urllib.request.urlretrieve(imgurls[i], fp)
            print("down done")
            # img=cv2.imread(fp)
            # print(img.shape)
            rr=model.run_from_filepath(fp)
            # if 
            t=rr[-3:]
            print("hhh")
            print(t)
            
            print(len(t))
            t.reverse()
            print(t)
            corrfac=0
            hflag=0
            for i in range(0,len(t),1):
                print(t[i][0])
                tt=t[i][0]
                if cropname in tt:
                    corrfac+=1
                if "health" in tt:
                    hflag+=1

            for i in range(0,len(t),1):
                # print(t[i][0])
                t[i][0]=t[i][0].split("___")[-1]

            print(corrfac,hflag)
            try:
                os.remove(fp)
            except Exception as e:
                print("file Not found")    
                
            if corrfac>=1 and t[0][0]!="None" and hflag<2 and t[0][1]>98.5 and ("health" not in t[0][0]):
                
                return {"status":"unhealthy", "diseases":t}

            elif corrfac>=1 and t[0][0]!="None" and t[0][1]>=98.5 and ("health" in t[0][0]): 
                
                return {"status":"healthy", "diseases":t}

            elif corrfac>=2 and t[0][0]!="None" and hflag>=1 and t[0][1]>=80:
                return {"status":"healthy","diseases":t}
            elif corrfac>=2 and t[0][0]!="None" and hflag<2:
                return {"status":"unhealthy", "diseases":t}

            else:
                return {"Unable to confirm with surety":t}
                
        except Exception as exc:
            print(exc)
            errstr="Error in image"
            os.remove(fp) 
            return errstr

@app.route('/keyword', methods=['POST'])

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
    

    
if __name__ == '__main__':
    app.run(debug=True)