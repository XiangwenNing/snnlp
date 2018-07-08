#encoding = utf-8
from TextCNN import TextCNN
from sklearn.model_selection import train_test_split
from data_helper2 import load_data_and_labels
import numpy as np
from sklearn.model_selection import train_test_split
def train():
    BATCH_SIZE = 32
    EPOCHS = 10
    EMBEDDING_SIZE = 256
    NUM_FILTERS = 128
    FILTER_SIZES = [3, 4, 5]
    # Load data and labels
    data,labels,num_words=load_data_and_labels("/erp/CLOUD_DISK/notebook/Me/taxCode.txt")
    indices=np.arange(data.shape[0])
    np.random.shuffle(indices)
    data=data[indices]
    labels=labels[indices]
    num_classes=len(set(labels))
    # split the data into a training set and a test set
    x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.3,random_state=42)
    print('Start training.')
    #Define TextCNN model
    text_cnn = TextCNN(num_class=num_classes,
                       num_words=num_words,
                       sequence_length=data.shape[1],
                       embedding_size=EMBEDDING_SIZE,
                       num_filters=NUM_FILTERS,
                       filter_sizes=FILTER_SIZES)
    model = text_cnn.model
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # evaluate the model
    score = model.evaluate(x_test,y_test, verbose=0)
if  __name__ == '__main__':
    train()