import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, debug=False):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        self.debug = debug
        print('iterations:', iterations)
        print('learning_rate:', learning_rate)
        print('hidden_nodes:', hidden_nodes)
        print('output_nodes:', output_nodes)
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(x):
            return  1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        
        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''

        ### Forward pass ###
        # Hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        # (There is no activation function in final layer because is regression )
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''

        #### Implement the backward pass here ####
        
        inputs = np.array(X, ndmin=2) 
        if self.debug:
            print('inputs', inputs.shape)
        
        # Output error 
        error = np.array(y - final_outputs, ndmin=2)

        if self.debug:
            print('error', error)
            
        # Calculate the gradient of the output activation wrt the input weights
        sig_prime = hidden_outputs * (1 - hidden_outputs)          # (dAh/dh)                
        hidden_grad = self.weights_hidden_to_output.T * sig_prime  # chain rule (dAh/dh * do/dAh * 1)       
        i_to_h_scores_prime = inputs.T
        W_i_to_h_prime = np.dot(i_to_h_scores_prime, hidden_grad)    # chain rule (dh/dWih * dAh/dh * do/dAh * 1) 
        if self.debug:
            print('hidden_outputs', hidden_outputs.shape)
            print('sig_prime  (dAh/dh):', sig_prime.shape)
            print('self.weights_hidden_to_output.T', self.weights_hidden_to_output.T.shape)
            print('hidden_grad (dAh/dh * do/dAh * 1)', hidden_grad.shape)
            print('i_to_h_scores_prime', i_to_h_scores_prime.shape)
            print('W_i_to_h_prime  (dh/dWih * dAh/dh * do/dAh * 1)', W_i_to_h_prime.shape) 

        # Calculate the gradient of the loss wrt the output layer weights
        h_to_o_scores_prime = np.array(hidden_outputs, ndmin=2).T
        W_h_to_o_prime = h_to_o_scores_prime    # same because the output activation is 1
        if self.debug:
            print( 'W_h_to_o_prime', W_h_to_o_prime.shape)
            
        # Backpropagate error        
        W_h_to_o_error_term =  error * W_h_to_o_prime
        W_i_to_h_error_term =  error * W_i_to_h_prime
            
        if self.debug:
            print( 'W_h_to_o_error_term', W_h_to_o_error_term.shape)
            print( 'W_i_to_h_error_term', W_i_to_h_error_term.shape)
            self.debug = False

        # TODO: Update the weights
        delta_weights_h_o += W_h_to_o_error_term # update hidden-to-output weights with gradient descent step        
        delta_weights_i_h += W_i_to_h_error_term # update input-to-hidden weights with gradient descent step
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records 
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

            
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        if self.debug:
            print('features', features.shape, features)
            print('*self.weights_input_to_hidden', self.weights_input_to_hidden.shape, self.weights_input_to_hidden)

            
        #### Implement the forward pass here ####
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)      # signals from hidden layer

        if self.debug:
            print('hidden_outputs', hidden_outputs.shape, hidden_outputs)
            print('self.weights_hidden_to_output', self.weights_hidden_to_output.shape, self.weights_hidden_to_output)

        # TODO: Output layer
        final_inputs =  np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        if self.debug:
            print('final_outputs', final_outputs)
            self.debug = False
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000 # epochs
learning_rate = 0.5
hidden_nodes = 6
output_nodes = 1
