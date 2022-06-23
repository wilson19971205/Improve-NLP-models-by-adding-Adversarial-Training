# CS388 Fa21 - Natural Language Processing Final Project, Hsuan-Wei Chen, EID: hc29434
import numpy as np
import json
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm

# ------------ Choose Datasets ------------
#iflytek data imformation
num_classes = 119
maxlen = 128
batch_size = 32

#tnews data imformation
#num_classes = 15
#maxlen = 64
#batch_size = 16
# -----------------------------------------

def load_data(filename):
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D

# ------------ Choose Datasets ------------
#iflytek
train_data = load_data('/iflytek/train.json')
valid_data = load_data('/iflytek/dev.json')

#iflytek + Adversarial sets
#train_data = load_data('/iflytek/train_ad.json')
#valid_data = load_data('/iflytek/dev_ad.json')

#tnews
#train_data = load_data('/tnews/train.json')
#valid_data = load_data('/tnews/dev.json')

#tnews + Adversarial sets
#train_data = load_data('/tnews/train_ad.json')
#valid_data = load_data('/tnews/dev_ad.json')
# -----------------------------------------

tokenizer = Tokenizer('/bert/vocab.txt', do_lower_case=True)

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

bert = build_transformer_model(config_path='/bert/bert_config.json',checkpoint_path='/bert/bert_model.ckpt',return_keras_model=False,)
output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(units=num_classes,activation='softmax',kernel_initializer=bert.initializer)(output)
model = keras.models.Model(bert.model.input, output)

def sparse_categorical_crossentropy(y_true, y_pred):
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)

def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values**2)
    return loss + 0.5 * epsilon * gp

def adversarial_training(model, embedding_name, epsilon=1):
    if model.train_function is None:
        model._make_train_function()
    old_train_function = model.train_function

    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    embeddings = embedding_layer.embeddings
    gradients = K.gradients(model.total_loss, [embeddings])
    gradients = K.zeros_like(embeddings) + gradients[0]

    inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    embedding_gradients = K.function(inputs=inputs,outputs=[gradients],name='embedding_gradients')

    def train_function(inputs):
        grads = embedding_gradients(inputs)[0]
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)
        K.set_value(embeddings, K.eval(embeddings) + delta)
        outputs = old_train_function(inputs)
        K.set_value(embeddings, K.eval(embeddings) - delta)
        return outputs

    model.train_function = train_function

# ------------- Adversarial training methods------------- 
# FGM
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(2e-5),metrics=['sparse_categorical_accuracy'])
adversarial_training(model, 'Embedding-Token', 0.5)

# gradient penalty
#model.compile(loss=loss_with_gradient_penalty,optimizer=Adam(2e-5),metrics=['sparse_categorical_accuracy'])
# -------------------------- ------------- _------------- 

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(u'val_acc: %.4f, best_val_acc: %.4f\n' %(val_acc, self.best_val_acc))


def predict_to_file(in_file, out_file):
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_generator.forfit(),steps_per_epoch=len(train_generator),epochs=50,callbacks=[evaluator])
else:
    model.load_weights('best_model.weights')
    # predict_to_file('/iflytek/test.json', 'iflytek_predict.json')
    # predict_to_file('/tnews/test.json', 'tnews_predict.json')