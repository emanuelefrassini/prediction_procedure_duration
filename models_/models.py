import keras
import numpy as np
from keras.layers import Layer,LSTM, Dropout, Dense,Input, multiply, concatenate, Activation, Masking, Reshape,Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Model
from keras import layers
import tensorflow.keras.backend as K
######################SMAPE##################################
def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum( np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

#################### LSTM ################
def model_LSTM(UNITS,DROPOUT_RATE,LEARNING_RATE,EPOCHS=1000):
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=-1))
    
    model.add((LSTM(UNITS, return_sequences=True, stateful=False)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='mae',  # weighted_categorical_crossentropy(weights),
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics='mae')
    
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(
        2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(2, np.round(EPOCHS/20)))

    model_callbacks = [early_stopping, lr_reduction]
    
    return model,model_callbacks
#################### Attention LSTM ################
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
      #  context = K.sum(context, axis=1)
        return context
def model_LSTM_Attention(UNITS,DROPOUT_RATE,LEARNING_RATE,ATTENTION_UNITS,EPOCHS=1000):
    model = keras.Sequential()
    model.add(keras.layers.Masking(mask_value=-1))
    
    model.add((LSTM(UNITS, return_sequences=True, stateful=False)))
    model.add(attention())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='mae',  # weighted_categorical_crossentropy(weights),
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics='mae')
    
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(
        2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(2, np.round(EPOCHS/20)))

    model_callbacks = [early_stopping, lr_reduction]
    
    return model,model_callbacks

#################### LSTM - FCN ################
def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se
def model_LSTM_FCN(X_1,X_2,UNITS,DROPOUT_RATE,LEARNING_RATE,EPOCHS=1000):
    ip = Input(shape=(X_1,X_2))

    x = Masking()(ip)
    x = LSTM(UNITS,return_sequences=True)(x)
    
    #x = LSTM(8,return_sequences=True)(x)
    x = Dropout(DROPOUT_RATE)(x)
   # x = Dense(128)(x)  # Adding a dense layer to match the output shape of the LSTM

   # y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)
   # print(y.shape)

    y = Conv1D(256, 5, padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)
   # print(y.shape)

    y = Conv1D(128, 3, padding='causal', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
  #  print(y.shape)

   # y = GlobalAveragePooling1D()(y)
   # y = Flatten()(y)
    x = concatenate([x, y])
    

    out = Dense(1, activation="relu")(x)

    model = Model(ip, out)
    
    model.compile(loss='mae', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics='mae')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(
        2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(2, np.round(EPOCHS/20)))
    
    # self.model.stop_training = True
    # ,MyThresholdCallback(),MyThresholdCallback2()]
    model_callbacks = [early_stopping, lr_reduction]
    return model,model_callbacks



###################### INCEPTIONTIME  ######################################
def _inception_module(input_tensor,BATCH_SIZE,EPOCHS, stride, activation,
                 nb_filters, use_residual, depth, kernel_size,use_bottleneck,
                 bottleneck_size):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
       input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                            padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer( input_tensor, out_tensor):
    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def model_INCEPTIONTIME (X_1,X_2,LEARNING_RATE,EPOCHS,BATCH_SIZE,stride=1, activation='linear',
                 nb_filters=32, use_residual=True, depth=6, kernel_size=41,use_bottleneck=True,
                 bottleneck_size = 32):
    input_shape=(X_1,X_2)
    input_layer = keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x,BATCH_SIZE,EPOCHS, stride, activation,
                         nb_filters, use_residual, depth, kernel_size,use_bottleneck,
                         bottleneck_size)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    #gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = Dense(1, activation="relu")(x)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='mae', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics='mae')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(
        2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(2, np.round(EPOCHS/20)))
    
    # self.model.stop_training = True
    # ,MyThresholdCallback(),MyThresholdCallback2()]
    model_callbacks = [early_stopping, lr_reduction]
    return model,model_callbacks

######################## TRANSFORMER ##############################################
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res
def model_transformer(X_1,X_2,LEARNING_RATE,EPOCHS,
    head_size,
    ff_dim,
    num_transformer_blocks,
    num_heads=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
        ):
    inputs = keras.Input(shape=(X_1,X_2))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="relu")(x)
    model = Model(inputs, outputs)

    model.compile(loss='mae', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics='mae')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=max(
        2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=max(2, np.round(EPOCHS/20)))
    
    # self.model.stop_training = True
    # ,MyThresholdCallback(),MyThresholdCallback2()]
    model_callbacks = [early_stopping, lr_reduction]
    return model,model_callbacks

