import tensorflow as tf
from src.pipelines import parse_json
from src.pipelines import data_gen
import numpy as np
import random
import sys
import os
from src.utils import normalize, tokenize


class tensorflow_cnn:

    def __init__(self, trainingdata, mode = 'load', model_path='', vocab={}, doc_length=500, batch_size=32, rand_seed = 41, \
                 hidden_layer_len = 32, connected_layer_len = 600, learning_rate = 0.001, num_steps=100, print_step=10, **kwargs):
        self.trainingdata = trainingdata
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if len(vocab)==0:
            print("Error: Must pass in a vocabulary with at least one entry", file=sys.stderr)
            quit()
        elif len(vocab)%2!=0:
            # We are going to add a bottom item as Tensorflow requires an even array
            # This doesn't need to be a real word, in fact it is better that it is not
            vocab['rwehwerohfaiuhfrahoiraehwiu'] = len(vocab)

        self.vocab_dict = vocab
        self.doc_length = doc_length
        print("Starting to initialize the CNN", file=sys.stderr)

        hot_docs, hot_clusters, n_classes = self.prep_news_data(self.vocab_dict, self.doc_length, self.doc_length)
        #grab a label randomly for when we fit a document - it doesn't matter what this is...
        self.hot_fake_label = hot_clusters[0]
        #set up the elements of the tensorflow model
        n_hidden = hidden_layer_len
        n_hidden_full=connected_layer_len
        n_characters = len(self.vocab_dict)
        n_half_characters = int(n_characters/2)
        n_char_length = self.doc_length
        learning_rate = learning_rate
        num_steps = num_steps
        print_step = print_step
        connected_layer_dim1 = int(n_characters/2.0*n_char_length/2.0*n_hidden)
        batch_size = batch_size
        random.seed(rand_seed)

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, n_characters, n_char_length])

        self.y = tf.placeholder(tf.float32, [None, n_classes])

        #weights and biases for the layers
        W_1 = tf.Variable(tf.truncated_normal(shape=[n_characters, 5, 1,n_hidden]))
        b_1 = tf.Variable(tf.constant(0.1, shape=[n_hidden]))

        W_connect = tf.Variable(tf.truncated_normal(shape=[connected_layer_dim1,n_hidden_full]))
        b_connect  = tf.Variable(tf.constant(0.1, shape=[n_hidden_full]))

        W_out = tf.Variable(tf.truncated_normal(shape=[n_hidden_full, n_classes]))
        b_out = tf.Variable(tf.constant(0.1, shape=[n_classes]))


        def build_cnn(x_in):
            x_shaped = tf.reshape(x_in, [-1, n_characters, n_char_length, 1])
            #print(x_shaped)
            lay1 = tf.nn.conv2d(x_shaped, W_1, strides = [1,1,1,1], padding='SAME', name="conv1")
            #print(lay1)
            hidden1 = tf.nn.relu(lay1+b_1)
            #print(hidden1)
            pool1 = tf.nn.max_pool(hidden1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name="pool1")
            #print(pool1)

            pool1_flat = tf.reshape(pool1, [-1, connected_layer_dim1]) #TODO get the shape in here better :)
            #print(pool1_flat)
            connected = tf.nn.relu(tf.matmul(pool1_flat, W_connect) + b_connect, name="connected")
            #print(connected)

            out_layer = tf.matmul(connected,W_out) + b_out
            #print(out_layer)
            return out_layer, connected

        pred, self.connected_layer = build_cnn(self.x)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        session_file_path = os.path.join(model_path, "tf_trained_session.cpt")

        if mode=='load':
            print('Loading the CNN Model', file=sys.stderr)
            saver = tf.train.Saver({"my_w1": W_1, "my_wcon": W_connect,
                                   "my_b1": b_1, "my_bcon": b_connect})
            init = tf.initialize_all_variables()
            sess = tf.Session()
            # Restore variables from disk.
            saver.restore(sess, session_file_path)

            self.session = sess
        else:
            print('Training the CNN Model', file=sys.stderr)
            # Initializing the variables
            init = tf.initialize_all_variables()

            # Add weights and biases to the be saved
            # We don't actually have to save the out variables as we are interested in the connected layer only
            # This is good because otherwise Tensorflow errors out because of the model's sizes (appears the model must be less than 2gb)
            saver = tf.train.Saver({"my_w1": W_1, "my_wcon": W_connect,
                                   "my_b1": b_1, "my_bcon": b_connect})

            sess = tf.Session()
            sess.run(init)
            step = 1
            print("Building Tensorflow CNN", file=sys.stderr)

            while step < num_steps:

                batch_doc, batch_label = self.generate_batch(hot_docs, hot_clusters, batch_size)

                sess.run(optimizer, feed_dict={self.x: batch_doc, self.y: batch_label})
                if step % print_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={self.x: batch_doc, self.y: batch_label})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={self.x: batch_doc, self.y: batch_label})
                    print("Iter " + str(step) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))
                step += 1

            self.session = sess
            print("Tensorflow CNN trained", file=sys.stderr)


            print("Saving the session", file=sys.stderr)
            saver.save(sess, session_file_path)
            print("Session Saved", file=sys.stderr)


    def transform_doc(self, doc, corpus):

        hot_doc = data_gen.run_onehot(doc, self.vocab_dict, self.doc_length, self.doc_length)
        hot_corpus = data_gen.run_onehot(corpus, self.vocab_dict, self.doc_length, self.doc_length)

        hot_docs = []
        hot_docs.append(hot_doc)
        hot_docs.append(hot_corpus)
        fake_labels = []
        fake_labels.append(self.hot_fake_label)
        fake_labels.append(self.hot_fake_label)
        #run the network for both docs as once as the input expects a list and it is likely faster
        #the labels aren't used so it doesn't matter what those are
        connected_layers = self.session.run(self.connected_layer, feed_dict={self.x: hot_docs, self.y: fake_labels})
        return connected_layers

    def prep_news_data(self, vocab, min_length, max_length):
        from sklearn.datasets import fetch_20newsgroups
        newsgroups= fetch_20newsgroups()

        documents = [data_gen.run_onehot(normalize.xml_normalize(text), vocab, min_length, max_length)
                     for text in newsgroups.data]
        labels = newsgroups.target

        #encode the labels in a dictionary
        unique_labels = np.unique(labels)
        i = 0
        unique_label_dict = {}
        for u_c in unique_labels:
            unique_label_dict[u_c] = i
            i +=1

        hot_labels = []
        n_classes = len(unique_labels)
        for c in labels:
            cluster_vect = np.zeros(n_classes, dtype=int)
            cluster_vect[unique_label_dict[c]]=1
            hot_labels.append(cluster_vect.tolist())

        return documents, hot_labels, n_classes

    def prep_data(self, in_data, vocab, min_length, max_length):
        documents = []
        labels = []

        for entry in in_data:
            text = entry["body_text"]
            documents.append(data_gen.run_onehot(text, vocab, min_length, max_length))
            label = entry["cluster_id"]
            labels.append(label)

        #encode the labels in a dictionary
        unique_labels = np.unique(labels)
        i = 0
        unique_label_dict = {}
        for u_c in unique_labels:
            unique_label_dict[u_c] = i
            i +=1

        hot_labels = []
        n_classes = len(unique_labels)
        for c in labels:
            cluster_vect = np.zeros(n_classes, dtype=int)
            cluster_vect[unique_label_dict[c]]=1
            hot_labels.append(cluster_vect.tolist())

        return documents, hot_labels, n_classes

    def generate_batch(self, documents, labels, batch_size):
        if batch_size>len(documents):
            return documents, labels

        #first shuffle the documents
        combined = list(zip(documents, labels))
        random.shuffle(combined)
        documents[:], labels[:] = zip(*combined)

        return documents[:batch_size], labels[:batch_size]