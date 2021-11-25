import os
import sys
import scipy.io
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import pprint



def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content
    """
    # Retrieve dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # compute the cost with tensorflow
    J_content = 1/(4 * n_H * n_W * n_C) * (tf.reduce_sum(tf.square(tf.subtract(a_C, a_G, name='sub'), name='square'), name='sum'))
  
    return J_content



def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, A, transpose_b = True, name=None)
    
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer 
    """
    
    # Retrieve dimensions from a_G 
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) 
    a_S = tf.reshape(a_S, shape=[n_H * n_W, n_C], name=None)
    a_S = tf.transpose(a_S)
    a_G = tf.reshape(a_G, shape=[n_H * n_W, n_C], name=None)
    a_G = tf.transpose(a_G)

    # Computing gram_matrices for both images S and G 
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss 
    J_style_layer = ((.5 / (n_H * n_W * n_C)) ** 2)*(  tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(GS, GG))))   )
    
    return J_style_layer



def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style 
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    
    return J


def model_nn(sess, input_image, num_iterations = 10):
    
    tf.disable_eager_execution() 
    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)
        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # Print every 20 iteration.
        if i%2 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image



if __name__ == '__main__':

    ''' Here starts the interactive part '''
    
    tf.disable_eager_execution() 
    # Reset the graph
    tf.reset_default_graph()
    # Start interactive session
    sess = tf.InteractiveSession()

    content_image = imageio.imread("images/profilbild-1.tif")
    content_image = reshape_and_normalize_image(content_image)

    style_image = imageio.imread("images/monet.jpg")
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)

    # Loading the vgg-19 model
    model = load_vgg_model("imagenet-vgg-verydeep-19.mat")

    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model e.g.['conv4_2'] 
    # and isn't evaluated yet. Later in the code, the image G is assigned as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)


    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))

    STYLE_LAYERS = [
        ('conv1_1', 0.2),
        ('conv2_1', 0.2),
        ('conv3_1', 0.2),
        ('conv4_1', 0.2),
        ('conv5_1', 0.2)]

    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS)


    J = total_cost(J_content, J_style, alpha=10, beta=40)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step 
    train_step = optimizer.minimize(J)


    model_nn(sess, generated_image)
