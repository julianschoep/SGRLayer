"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

from utils import preprocessing
import numpy as np
import horovod.tensorflow as hvd
import os


import memory_saving_gradients
# monkey patch tf.gradients to point to our custom version, with automatic checkpoint selection
tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

# from SGR import SGRLayer

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4

M = 204  # no of symbolic entities in knowledge graph
Dc = 256  # symbolic feature length
Dl = 256  # input layer depth
K = 300  # fasttext embedding vector size



def histogram(name, scalar):
    # TF summary histogram wrapper to see how the various layers in the SGR progress.

    def clip_fn(t,min_val, max_val):
        return tf.clip_by_value(t, min_val, max_val)
    clip_dict = {
        'input_features':(0,10),
        'output_features':(-2,2),
        'applied_mapping':(-50,50),
        'compatibility':(-10,60),
        'evolved_feat':(-10,10),
        'encoded_concat_feat':(-150,150),
        'visual_feat':(0,250)
    }
    
    if 'feat' in name:
        tf.summary.histogram(name+'_clip50',clip_fn(scalar,-50,50))
        tf.summary.histogram(name+'_clip10',clip_fn(scalar,-10,10))
    if 'softmax' in name:
        tf.summary.histogram(name+'_clip.001', clip_fn(scalar,0,0.001))
        tf.summary.histogram(name+'_clip.01', clip_fn(scalar,0,0.01))
        tf.summary.histogram(name+'_clip.4', clip_fn(scalar,0,0.4))

    else:#name in zooms:
        tf.summary.histogram(name+'_clip1', clip_fn(scalar,-2,2))
        tf.summary.histogram(name+'_clip1', clip_fn(scalar,-1,1))
        tf.summary.histogram(name+'_clip5', clip_fn(scalar,-5,5))

    return tf.summary.histogram(name, scalar)


def row_normalize(mat):
    Q = tf.reduce_sum(mat, axis=-1)  # take sum along rows # input mat 1,H,W # 33x33
    Q = tf.cast(Q, tf.float32)  # conver tot float
    # take max with espilon to avoid inf/nan in rsqrt
    # _EPS = tf.constant(1e-7)
    # Q = tf.map_fn(lambda x: tf.maximum(_EPS, x), Q)
    # Inverse sqrt reciprocal (i.e. negative half power)
    sQ = tf.rsqrt(Q)
    sQ = tf.matrix_diag(sQ)  # make diagonal mat
    # do symmetric normalization
    return tf.matmul(sQ, tf.matmul(mat, sQ, name='right_side_norm'), name='left_side_norm')


def norm_adjacency(adjacency_matrix):
    # Adds identity connection to adjacency matrix, converts to tf constant
    # and normalize

    # Adjecency matrix can be of any shape.
    a = adjacency_matrix  # MxM
    assert a.shape[0] == a.shape[1], 'adjacency matrix should be symmetirc'
    # row normalize the adjacency matrix
    a += np.identity(a.shape[0])  # add identity connections
    tf_adj = tf.constant(a, dtype=tf.float32)
    norm_adj = row_normalize(tf_adj)
    return norm_adj


def SGRLayer(inputs, adj_matrix, embeddings, batch_norm_decay, is_training):
    histogram('SGR_input_features', inputs)
    to_inspect = {} # layer activations that we want to visualize later
    with tf.variable_scope('SGR', reuse=tf.AUTO_REUSE):

        # For the experiment without using fasttext embeddings.
        _fasttext_embeddings = tf.Variable(tf.random_normal( 
                                (M,K),
                                mean=-0.0061,
                                stddev=0.223,
                                dtype=tf.float32
                            ),dtype=tf.float32,
                              trainable=False)

        fasttext_embeddings = tf.Variable(embeddings, dtype=tf.float32,trainable=False,name='fasttext_embeddings')
        norm_adj = norm_adjacency(adj_matrix)

        INPUT_SHAPE = tf.shape(inputs)#
       
        x = inputs

        ### LOCAL TO SEMANTIC MODULE ###
        with tf.variable_scope('voting'):
            tf.logging.info('Building voting module')

            with tf.variable_scope('calc_votes'):
                
                # will have shape [?,H,W,M]
                votes = layers_lib.conv2d( # 
                    inputs, M, (1, 1), stride=1, 
                    normalizer_fn=tf.nn.softmax,
                    normalizer_params={'axis':-1, 'name':'voting_softmax_target'} ,
                    activation_fn=None, # NO RELU IS USED IN THE PAPER HERE
                    scope='L2S_Wa') # uses Xavier initialization

                to_inspect['softmax_vote_activations'] = votes

                # shape [?,H,W,M]
                histogram('vote_softmax', votes)
                votes = tf.reshape(
                    votes, [INPUT_SHAPE[0], INPUT_SHAPE[1] * INPUT_SHAPE[2], M])  # shape [?,H*W,M]
                votes = tf.transpose(votes, [0, 2, 1])  # shape [?,M, H*W] # every pixel Xi distributes
                # voting over the M nodes. np.sum(0,:,0]) must be 1

            with tf.variable_scope('assign_votes'):
                in_feat = layers_lib.conv2d( # transform each local feature into length Dc
                    inputs, Dc, (1, 1), stride=1, 
                    biases_initializer=None,
                    weights_initializer=tf.glorot_normal_initializer(), 
                    activation_fn=None, # NO RELU IS USED IN THE PAPER HERE
                    scope='L2S_Wps')  # shape [?,H,W,Dc]
                activation_summary(in_feat)
                # shape [H*W,Dc]
                in_feat = tf.reshape(in_feat, [
                                     INPUT_SHAPE[0], INPUT_SHAPE[1] * INPUT_SHAPE[2], Dc])  # shape [?,H*W,Dc]
                in_feat = tf.matmul(
                    votes, in_feat, name='applied_votes')  # shape [?,M,Dc]
                visual_features = tf.nn.relu(
                    in_feat, name='visual_feat')  # shape [?,M,Dc] This relu is not mentioned in the text, but is shown
                                                  # in figure 2 in the paper.
                histogram('visual_features', visual_features)
                to_inspect['visual_features'] = visual_features

        ### GRAPH REASONING MODULE ###
        with tf.variable_scope('reasoning'):
            tf.logging.info('Building reasoning module')

            # Fasttext embeddings should be externally supplied
            # Fasttext embedding has shape [M,K], tile em for the features each batch slice
            
            fasttext_embeddings = tf.expand_dims(fasttext_embeddings, 0)
                                        
            fasttext_embeddings = tf.tile(
                fasttext_embeddings, [INPUT_SHAPE[0], 1, 1])  # sh [b,M,K]

            concat_feat = tf.concat([visual_features, fasttext_embeddings],
                                    axis=-1, name='concat_embed')  # shape [b,M,Dc+K]
            tf.summary.histogram('concat_features', concat_feat)
            to_inspect['embed_concat_feat'] = concat_feat
            activation_summary(concat_feat)
            

            # Encode into dc length vector (e.g. [M,(Dc+k)] to [M,(Dc)]))
            # transformed_feat = tf.layers.conv2d(inputs=concat_feat,
            # filters=Dc,kernel_size=(1),name='encode_concat_feats') # encode
            # back to shape [M,Dc]
            # This operation without RELU somehow forces the features to become very negative, which causes 
            # the output of the SGR layer to be zero due to the RELU activation at the end. Lets try to put a bias 
            # and relu activation here to force the output to be positive.
            # concat_feat has shape [b,M,Dc+K]
            # make shape [b,M,Dc]
            with tf.variable_scope('encode_concat'):
                concat_feat = tf.reshape(concat_feat,[-1,Dc+K]) # shape [b*M,Dc+K]
                Wg= tf.get_variable(name="Wg", dtype=tf.float32,
                                          shape=(Dc+K, Dc),
                                          initializer=tf.glorot_normal_initializer(),
                                          trainable=True)
                tf.summary.histogram('Wg_encode_concat', Wg)
                transformed_concat_feat = tf.matmul(concat_feat,Wg)# shape [b*M,Dc] @ square weight to force positive
                transformed_concat_feat = tf.reshape(transformed_concat_feat,[INPUT_SHAPE[0],M, Dc])

            histogram('encoded_concat_feat', transformed_concat_feat)
            
            tf.logging.info('Creating second log module')
            to_inspect['embed_transformed_feat'] = transformed_concat_feat
            
            # norm_adj has shape MxM
            # evolved feat has shape [b,M,Dc]
            # Approach one: Mulitply with row normalized adjacency matrix, for every batch
            # Approach two: Stack batch features along columns
            #  multiply norm_adj with this [Mxb*Dc] tensor. 
            # The resulting tensor will be the 'evolved' batches still stacked along the cols.
            # Reshape back to stack batches back along the first dimension.
            with tf.variable_scope('graph_convolution'):
                tile_norm_adj = tf.tile(norm_adj[tf.newaxis,:,:],[INPUT_SHAPE[0],1,1]) # shape [b,M,M] @ [b,M,Dc] --> [b,M,Dc] # approach two
                evolved_feat = tf.matmul(tile_norm_adj,transformed_concat_feat,name='matmul_adj_feat') # [b,M,Dc]
                #evolved_feat = tf.map_fn(lambda vis_feat: tf.matmul(norm_adj, vis_feat),transformed_feat, name='matmul_adj_feat') # shape [b,M,Dc] # approach one
                histogram('evolved_feat',evolved_feat)

            to_inspect['embed_evolved_feat'] = evolved_feat

            histogram('evolved_feat_before_relu', evolved_feat)
            evolved_feat = tf.nn.relu(
                evolved_feat, name='evolved_feats')  # shape [b,M,Dc]

            histogram('evolved_feat', evolved_feat)
        ## SEMANTIC TO LOCAL MODULE ###
        with tf.variable_scope('mapping'):
            tf.logging.info('Building mapping module')
            with tf.variable_scope('calc_mapping'):
                tf.logging.info('Setup calc mapping graph')
                
                # use alternative that uses nested map_fn functions.
                # VERIFIED Works. Has memory of 950mb, still doesnt enable OS8, but alternatives are difficult
                # to do without nested map_fns, which in turn have some gradient computation errors (Nonetype has no attribute 'op')
                # This seems like a good middle ground between memory usage and speed, and it works

                def _compute_compat_batches(inp,evolved):
                    batch_data = (inp,evolved)
                    def compute_compat_batch(batch_x, batch_feat):
                        # batch_x has shape [H,W,Dl]
                        # batch_feat has shape [M,Dc]
                        sh = tf.shape(batch_x)
                        batch_x = tf.reshape(batch_x,[sh[0]*sh[1],sh[2]]) # shape [H*W,Dl]
                        batch_x = tf.tile(tf.expand_dims(batch_x,1),[1,M,1]) # has shape [H*W,M,Dl]

                        batch_feat = tf.tile(tf.expand_dims(batch_feat,0),[sh[0]*sh[1],1,1]) # sh [H*W,M,Dc]

                        compare_concat = tf.concat([batch_feat, batch_x], axis=-1,
                                               name='concat_inp')  # (H*WxMx(Dc+Dl))
                        compare_concat.set_shape([None, None, Dc+Dc])
                        compat = layers_lib.conv2d( # Ws in paper
                                        compare_concat[tf.newaxis,:,:,:],
                                        1, (1, 1), 
                                        stride=1, 
                                        biases_initializer=None,
                                        activation_fn=None,
                                        scope='Ws') # has shape (H*W,M,1)
                        
                       return tf.reshape(compat,[sh[0]*sh[1],M])

                    return tf.map_fn(lambda batch: compute_compat_batch(*batch), batch_data,dtype=tf.float32)
                                    
                compat = _compute_compat_batches(inputs, evolved_feat) # will have shape [b,H*W,M]
                tf.summary.scalar('compat/sparsity', tf.nn.zero_fraction(compat))
                activation_summary(compat)
                histogram('compatibility',compat)
                to_inspect['compatibility'] = compat
                
                mapping = tf.nn.softmax(
                    compat, axis=1, name='mapping_softmax')  # (b,H*WxM) ! note axis, we want to take softmax w.r.t. other pixels,
                tf.summary.scalar('mapping/sparsity', tf.nn.zero_fraction(mapping))
   
                to_inspect['mapping_target'] = mapping
                histogram('mapping_softmax', mapping)

                # NOT over other visual nodes. So a visual feature gets distributed to output nodes according to how compatible its value is
                # w.r.t. to other pixels, 

                # They say they do this in the paper? Sec. 3.4. doing symmetric row normalization of possibly asymetric (or even 3d?) matrix
                # within the computational graph of tensorflow is difficult. 
                # Not shown in figure 2, but implemented anyway. 
                
                # Instead we use tf build int normalization function. 
                # Faster than our own implementation and supports arbitrareliy sized tensors, which is useful for eval.
                with tf.variable_scope('row_normalize_mapping'):  
                    #mapping has shape [b,H*W,M]
                    mapping_norm = tf.norm(mapping,ord=1,axis=-1) # has shape [b,H*W] holding sums along M dim
                    mapping_norm = tf.tile(tf.expand_dims(mapping_norm,axis=-1),[1,1,M]) # copy sums along M dim for element wise divide shape [b,H*W,M]
                    mapping = mapping / mapping_norm # l1 normalized mapping, where the M dim sums up to 1.
                    #mapping = tf.nn.l2_normalize(mapping,axis=2) # does l2 normalization along the rows. 
                    histogram('mapping_softmax_row_norm', mapping)
                to_inspect['mapping_row_norm'] = mapping  # shape [b,H,W,M]


            with tf.variable_scope('apply_mapping'):
                tf.logging.info('Setup apply mapping graph')
                
                Wsp = tf.get_variable(name="Wsp", dtype=tf.float32,
                                      shape=(Dc, Dl),
                                      initializer=tf.glorot_normal_initializer(),
                                      trainable=True)

                tf.summary.histogram('W_transf_evolved', Wsp)
                # evolved_feat # shape [b,M,Dc]

                # merge batch with M again for efficient mult with Wsp
                evolved_feat = tf.reshape(evolved_feat,[-1,Dc]) # shape [b*M,Dc]

                transf_evolved = tf.matmul(evolved_feat,Wsp) #[b*M,Dl]
                transf_evolved = tf.reshape(transf_evolved,[INPUT_SHAPE[0],M,Dl]) # [b,M,Dl]

                #transf_evolved = tf.map_fn(lambda batch_evolved_feat: tf.matmul(batch_evolved_feat, Wsp),evolved_feat) # shape [b,M,Dl]
                activation_summary(transf_evolved)
                to_inspect['transf_evolved'] = transf_evolved
                # merge back
                histogram('transf_evolved', transf_evolved)

                # Not mentioned in paper, but before applying mapping, reshape
                # row-normalized Ag (e.g. mapping) back to H*W x M so we can matmul
                # it! 
                mapping = tf.reshape(
                    mapping, [INPUT_SHAPE[0], INPUT_SHAPE[1] * INPUT_SHAPE[2], M])  # shape b,H*WxM

                # Distribute evolved features to output feature map according to
                #   softmax mapping
                applied_mapping = tf.matmul(
                    mapping, transf_evolved, name='applied_mapping')  # bx(H*W)xDl
                to_inspect['applied_mapping_before_relu'] = applied_mapping
                histogram('applied_mapping_before_relu', applied_mapping)
                applied_mapping = tf.nn.relu(
                    applied_mapping, name='mapping_relu')  # bx(H*W)xDl
                applied_mapping = tf.reshape(applied_mapping, [INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2], Dl],
                                             name='mapped_feat_target')  # b,HxWxDl    # Look at feature maps! (there's a lot so maybe difficult)
                
                histogram('SGR_output_features', applied_mapping)
                to_inspect['applied_mapping'] = applied_mapping
                to_inspect['SGR_input'] = inputs
                tf.summary.scalar('applied_mapping/sparsity', tf.nn.zero_fraction(applied_mapping))
                
    return (inputs+applied_mapping),to_inspect

