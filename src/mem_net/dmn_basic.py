import random
import numpy as np

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import pickle

from src.mem_net import utils
from src.mem_net import nn_utils

floatX = theano.config.floatX


class DMN_basic:
    
    def __init__(self, train_raw, test_raw, word2vec, word_vector_size,
                dim, mode, answer_module, input_mask_mode, memory_hops, l2, 
                normalize_attention, **kwargs):

        print("==> not used params in DMN class:", kwargs.keys())
        self.vocab = {}
        self.ivocab = {}
        
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
        self.mode = mode
        self.answer_module = answer_module
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.l2 = l2
        self.normalize_attention = normalize_attention

        self.to_return = "word2vec" #TODO pass these in once updates are wrapped in
        self.encoder_decoder = None
        self.vocab_dict = {}
        
    # Process the input into its different parts and calculate the input mask
        self.train_input = train_raw['inputs']
        self.train_q = train_raw['questions']
        self.train_answer = train_raw['answers']
        self.train_input_mask=train_raw['input_masks']
        self.test_input = test_raw['inputs']
        self.test_q = test_raw['questions']
        self.test_answer = test_raw['answers']
        self.test_input_mask=test_raw['input_masks']
        # self.train_input, self.train_q, self.train_answer, self.train_input_mask = self._process_input(train_raw)
        # self.test_input, self.test_q, self.test_answer, self.test_input_mask = self._process_input(test_raw)
        self.vocab_size = 2 #vocab size might only be used for the answer module and we only want to give that two choices
        # print(self.train_answer[0])
        # print(self.train_input_mask[0])
        # print(np.array(self.train_input[0]).shape)

        # Get the size of the word_vector from the data itself as this can be variable now
        self.word_vector_size = np.array(self.train_input[0]).shape[1]

        self.input_var = T.matrix('input_var')
        self.q_var = T.matrix('question_var')
        self.answer_var = T.iscalar('answer_var')
        self.input_mask_var = T.ivector('input_mask_var')
        print(len(self.train_answer), sum(self.train_answer), len(self.test_answer), sum(self.test_answer))
        
            
        print("==> building input module")
        self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        inp_c_history, _ = theano.scan(fn=self.input_gru_step, 
                    sequences=self.input_var,
                    outputs_info=T.zeros_like(self.b_inp_hid))

        self.inp_c = inp_c_history.take(self.input_mask_var, axis=0)
        
        self.q_q, _ = theano.scan(fn = self.input_gru_step,
			sequences = self.q_var,
			outputs_info = T.zeros_like(self.b_inp_hid))

        self.q_q = self.q_q[-1]


        print("==> creating parameters for memory module")
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 2))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))
        

        print("==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops)
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))
        
        last_mem = memory[-1]
        
        print("==> building answer module")
        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))
        
        if self.answer_module == 'feedforward':
            self.prediction = nn_utils.softmax(T.dot(self.W_a, last_mem))
        
        elif self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
            def answer_step(prev_a, prev_y):
                a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
                
                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]
            
            # TODO: add conditional ending
            dummy = theano.shared(np.zeros((self.vocab_size, ), dtype=floatX))
            results, updates = theano.scan(fn=answer_step,
                outputs_info=[last_mem, T.zeros_like(dummy)],
                n_steps=1)
            self.prediction = results[1][-1]
        
        else:
            raise Exception("invalid answer_module")
        
        
        print("==> collecting all parameters")
        self.params = [self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                  self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                  self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid,
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid,
                  self.W_b, self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]
        
        if self.answer_module == 'recurrent':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
        
        
        print("==> building loss layer and computing updates")
        self.loss_ce = T.nnet.categorical_crossentropy(self.prediction.dimshuffle('x', 0), T.stack([self.answer_var]))[0]
        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
        
        updates = lasagne.updates.adadelta(self.loss, self.params)
        
        if self.mode == 'train':
            print("==> compiling train_fn")
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var], 
                                       outputs=[self.prediction, self.loss],
                                       updates=updates)
        
        print("==> compiling test_fn")
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var],
                                  outputs=[self.prediction, self.loss, self.inp_c, self.q_q, last_mem])
        
        
        if self.mode == 'train':
            print("==> computing gradients (for debugging)")
            gradient = T.grad(self.loss, self.params)
            self.get_gradient_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var], outputs=gradient)

    
    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper: 
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
        z = T.nnet.sigmoid(T.dot(W_upd_in, x) + T.dot(W_upd_hid, h) + b_upd)
        r = T.nnet.sigmoid(T.dot(W_res_in, x) + T.dot(W_res_hid, h) + b_res)
        _h = T.tanh(T.dot(W_hid_in, x) + r * T.dot(W_hid_hid, h) + b_hid)
        return z * h + (1 - z) * _h
    
    
    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    
    
    def new_attention_step(self, ct, prev_g, mem, q_q):
        cWq = T.stack([T.dot(T.dot(ct, self.W_b), q_q)])
        cWm = T.stack([T.dot(T.dot(ct, self.W_b), mem)])
        z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, T.abs_(ct - q_q), T.abs_(ct - mem), cWq, cWm])
        
        l_1 = T.dot(self.W_1, z) + self.b_1
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_2, l_1) + self.b_2
        G = T.nnet.sigmoid(l_2)[0]
        return G
        
        
    def new_episode_step(self, ct, g, prev_h):
        gru = self.GRU_update(prev_h, ct,
                             self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                             self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                             self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid)
        
        h = g * gru + (1 - g) * prev_h
        return h
       
    
    def new_episode(self, mem):
        g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0])) 
        
        if (self.normalize_attention):
            g = nn_utils.softmax(g)
        
        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))
        
        return e[-1]

    
    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'wb') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = 2
            )
    
    
    def load_state(self, file_name):
        print("==> loading state %s" % file_name)
        with open(file_name, 'rb') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)

    def _process_input(self, data_raw):
        '''
            This module processes the raw data input and grabs all the relevant sections and calculates the input_mask.

        Args:
            data_raw: raw data coming in from main class.
        Returns:
            inputs section, answers section, questions section, and input_masks as numpy arrays.
        '''
        inputs = []
        answers = []
        input_masks = []
        questions = []
        for x in data_raw:
            inp = x["C"].lower().split(' ')
            inp = [w for w in inp if len(w) > 0]
            q = x["Q"].lower().split(' ')
            q = [w for w in q if len(w) > 0]
            # Sentence punctuation delimiters
            punkt = ['.','?','!']

            problem=False
            # NOTE: here we assume the answer is one word!
            if self.input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32)) # Get the input_masks for the data
            elif self.input_mask_mode == 'sentence':
                sent_mask = np.array([index for index, w in enumerate(inp) if w in punkt], dtype=np.int32)
                input_masks.append(sent_mask)
                if(len(sent_mask)<1):
                    #Pass over the data if there is only one sentence as this will cause an error later
                    problem=True
            else:
                raise Exception("invalid input_mask_mode")
            if problem:
                print("Passing over data: ", x["C"])
                continue

            #Process the documents
            inp_vector = utils.process_sent(inp, word2vec=self.word2vec, vocab=self.vocab, ivocab=self.ivocab,
                                            word_vector_size=self.word_vector_size, to_return=self.to_return, silent=True,
                                            encoder_decoder=self.encoder_decoder, vocab_dict=self.vocab_dict)

            q_vector = utils.process_sent(q, word2vec=self.word2vec, vocab=self.vocab, ivocab=self.ivocab,
                                            word_vector_size=self.word_vector_size, to_return=self.to_return, silent=True,
                                            encoder_decoder=self.encoder_decoder, vocab_dict=self.vocab_dict)
            # Process the words from the input, answers, and questions to see what needs a new vector in word2vec.
            # inp_vector = [utils.process_word(word = w,
            #                             word2vec = self.word2vec,
            #                             vocab = self.vocab,
            #                             ivocab = self.ivocab,
            #                             word_vector_size = self.word_vector_size,
            #                             to_return = "word2vec", silent=True) for w in inp]
            #
            # q_vector = [utils.process_word(word = w,
		    	# 		word2vec = self.word2vec,
				# 	vocab = self.vocab,
				# 	ivocab = self.ivocab,
				# 	word_vector_size = self.word_vector_size,
				# 	to_return = "word2vec", silent=True) for w in q]
            inputs.append(np.vstack(inp_vector).astype(floatX))
            questions.append(np.vstack(q_vector).astype(floatX))
            answers.append(utils.process_word(word = x["A"],
                                            word2vec = self.word2vec, 
                                            vocab = self.vocab, 
                                            ivocab = self.ivocab, 
                                            word_vector_size = self.word_vector_size, 
                                            to_return = "bool"))


        
        return inputs, questions, answers, input_masks

    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            print(len(self.train_input))
            return len(self.train_input)
        elif (mode == 'test'):
            return len(self.test_input)
        else:
            raise Exception("unknown mode")
    
    
    def shuffle_train_set(self):
        print("==> Shuffling the train set")
        combined = list(zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask))
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = list(zip(*combined))
        
    
    def step(self, batch_index, mode):
        '''
            This function is one step in an epoch and will run a training or testing step depending on the parameter.

	    Args:
		batch_index (int): step number for the epoch
		mode (str): 'train' or 'test' based on the mode of 
	    Returns:
	        Dictionary of predictions, answers, loss, number skipped, and the normal and gradient parameters
	'''
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")
        
        if mode == "train":
            theano_fn = self.train_fn # Theano function set
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            input_masks = self.train_input_mask
        elif mode == "test":    
            theano_fn = self.test_fn 
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
        else:
            raise Exception("Invalid mode")
            
        inp = inputs[batch_index].astype(floatX)
        q = qs[batch_index].astype(floatX)
        ans = answers[batch_index]
        input_mask = input_masks[batch_index]
        #print(np.array(inp).shape, np.array(q).shape, np.array(ans).shape, np.array(input_mask).shape)

        skipped = 0
        grad_norm = float('NaN')
        
        if mode == 'train':
            gradient_value = self.get_gradient_fn(inp, q, ans, input_mask) # Get and calculate the gradient function
            grad_norm = np.max([utils.get_norm(x) for x in gradient_value])
            
            if (np.isnan(grad_norm)):
                print("==> gradient is nan at index %d." % batch_index)
                print("==> skipping")
                skipped = 1
        
        if skipped == 0:
            ret = theano_fn(inp, q, ans, input_mask) # Run the theano function
        else:
            ret = [-1, -1]
        
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": np.array([ret[0]]),
                "answers": np.array([ans]),
                "current_loss": ret[1],
                "skipped": skipped,
                "log": "pn: %.3f \t gn: %.3f" % (param_norm, grad_norm)
                }
        