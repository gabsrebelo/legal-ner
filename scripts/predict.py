from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input, model_from_json, load_model
# from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
# from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import pickle
import spacy

def load_sentences(file):
	with open(file,'r') as f:
		lines = f.readlines()
		sentences = [nlp_pt(l) for l in lines if l != '\n']
		sentences = [[str(w) for w in s] for s in sentences] #transforma cada sentenca em uma list<string> de palavras
		print('{} sentencas encontradas'.format(len(sentences)))
	return sentences

def load_word2idx():
	with open("./models/colab/v1/word2idx_v1_13581","rb") as word2idx_file:
		word2idx = pickle.load(word2idx_file)
		print("Word2idx carregado!")
	return word2idx

def save_predict_sentence(p, output_file):
	writer = open(output_file, mode='a')
	p = np.argmax(p, axis=-1)
	writer.write("Sentença #{}\n".format(i))
	writer.write("{:15}||{}\n".format("Word", "Prediction"))
	writer.write(30 * "=")
	writer.write("\n")
	for w, pred in zip(test_sentence, p[0]):
	    writer.write("{:15}: {:5}\n".format(w, tags[pred]))
	
	writer.write("\n")

def pad_sentence(sentence):
	return pad_sequences(sequences=[[word2idx.get(w, 0) for w in sentence]], padding="post", value=0, maxlen=max_len)

if __name__ == "__main__":	
	#init spacy
	nlp_pt = spacy.load("pt_core_news_sm")

	output_file = "predicted_sentences_v2_colab.txt"
	max_len = 500
	tags = ['I-LEGISLACAO', 'O', 'B-LEGISLACAO']

	sentences = load_sentences("./Coleta de sentencas/text_sentences_extracted_test.txt")
	word2idx = load_word2idx()

	#model
	loaded_model = load_model("./models/colab/v1/model_v1.h5",custom_objects={'CRF':CRF,'crf_loss':crf_loss,'crf_accuracy':crf_accuracy, 'crf_viterbi_accuracy':crf_viterbi_accuracy})
	loaded_model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_accuracy])
	
	i = 1
	for test_sentence in sentences:
		#preprocessing
		x_test_sent = pad_sentence(test_sentence)

		#predict
		pred = loaded_model.predict(np.array([x_test_sent[0]]))
		save_predict_sentence(pred, output_file)
		i += 1
	