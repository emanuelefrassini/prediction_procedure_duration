import keras
from keras.layers import LSTM, Dropout, Dense,Input, multiply, concatenate, Activation, Masking, Reshape,Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Model
import tensorflow as tf
from keras import layers
import numpy as np
from tensorflow.keras.callbacks import TensorBoard

def model_creation(X,Y,granularity,method,UNITS,DROPOUT_RATE,LEARNING_RATE,EPOCHS,loss,weights,folder_path_tensorboard,num_transf_blocks=8,head_size=4,ffd=4,dropout_enc=0.25):
    if granularity=='min':
        NUM_CLASSES=5
    elif granularity=='max':
        NUM_CLASSES=15
    elif granularity=='time_binary':
        NUM_CLASSES=7

    if granularity == 'max' or granularity=='min':
        if method=='lstm':
            def create_model():
                model = keras.Sequential()
                model.add(keras.layers.Masking(mask_value=0))
        
                model.add(LSTM(UNITS, return_sequences=True, stateful=False))
                model.add(Dropout(DROPOUT_RATE))
                model.add(Dense(NUM_CLASSES, activation='softmax'))
        
                model.compile(loss=loss(weights),
                              optimizer=tf.keras.optimizers.Adam(
                                  learning_rate=LEARNING_RATE),
                              metrics=[keras.metrics.categorical_accuracy])
        
                return model
        elif method=='transformer':
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
            def create_model(
                head_size=head_size,
                num_heads=4,
                ff_dim=ffd,
                num_transformer_blocks=num_transf_blocks,
                mlp_units=[128],
                mlp_dropout=0.4,
                dropout=dropout_enc,
                    ):
                inputs = keras.Input(shape=(X.shape[1],X.shape[2]))
                x = inputs
                for _ in range(num_transformer_blocks):
                    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
            
                #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
                for dim in mlp_units:
                    x = layers.Dense(dim, activation="relu")(x)
                    x = layers.Dropout(mlp_dropout)(x)
                outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
                model = Model(inputs, outputs)

                model.compile(loss=loss(weights),
                             optimizer=tf.keras.optimizers.Adam(
                                 learning_rate=LEARNING_RATE),
                             metrics=[keras.metrics.categorical_accuracy])
                return model
            
            
        elif method=='lstm-fcn':
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
            def create_model():
                ip = Input(shape=(X.shape[1],X.shape[2]))

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
                print(y.shape)

                y = Conv1D(256, 5, padding='causal', kernel_initializer='he_uniform')(y)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = squeeze_excite_block(y)
                print(y.shape)

                y = Conv1D(128, 3, padding='causal', kernel_initializer='he_uniform')(y)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                print(y.shape)

               # y = GlobalAveragePooling1D()(y)

               # y = Flatten()(y)
    #            print(y.shape)



                x = concatenate([x, y])
                

                out = Dense(NUM_CLASSES, activation='softmax')(x)

                model = Model(ip, out)
               # model.summary()
                model.compile(loss=loss(weights),
                             optimizer=tf.keras.optimizers.Adam(
                                 learning_rate=LEARNING_RATE),
                             metrics=[keras.metrics.categorical_accuracy])
                # add load model code here to fine-tune

                return model

        
        tensorboard_callback = TensorBoard(
            log_dir=folder_path_tensorboard, histogram_freq=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=max(
            2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
        lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, np.round(EPOCHS/20)))
       
        # self.model.stop_training = True
        # ,MyThresholdCallback(),MyThresholdCallback2()]
        model_callbacks = [tensorboard_callback, early_stopping, lr_reduction]
        # self.model.stop_training = True
        # ,MyThresholdCallback(),MyThresholdCallback2()]

    elif granularity == 'time_binary_regression':
        def create_model():
            model = keras.Sequential()
            model.add(keras.layers.Masking(mask_value=0))

            model.add(LSTM(UNITS, return_sequences=False, stateful=False))
            model.add(Dropout(DROPOUT_RATE))
            model.add(Dense(1, activation='relu'))

            model.compile(loss='mean_absolute_error',  # weighted_categorical_crossentropy(weights),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics='mae'
                          )

            return model
        tensorboard_callback = TensorBoard(
            log_dir=folder_path_tensorboard, histogram_freq=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=max(
            2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
        lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae', factor=0.5, patience=max(2, np.round(EPOCHS/20)))
        
        # self.model.stop_training = True
        # ,MyThresholdCallback(),MyThresholdCallback2()]
        model_callbacks = [tensorboard_callback, early_stopping, lr_reduction]

    elif granularity == 'time_binary':
       # class PadSequences(Layer):
        ##       super(PadSequences, self).__init__(**kwargs)
         #       self.maxlen = maxlen
         #       self.padding = padding
         #       self.truncating = truncating

           # def call(self, inputs):
            #    return tf.keras.utils.pad_sequences(inputs, padding=self.padding, maxlen=self.maxlen, truncating=self.truncating)
            #def call(self, inputs):
             #   return tf.keras.backend.map_fn(
              #      lambda x: tf.keras.preprocessing.sequence.pad_sequences(x[None, :], padding=self.padding, maxlen=self.maxlen, truncating=self.truncating)[0],
               #     inputs
                #    )
        if method == 'lstm':
            def create_model():
                model = keras.Sequential()
              #  model.add(PadSequences(maxlen=MAXLENPADDED)(keras.Input(shape=(None,))))

                model.add(keras.layers.Masking(mask_value=0))

                model.add(LSTM(UNITS, return_sequences=True, stateful=False))
                model.add(LSTM(UNITS, return_sequences=True, stateful=False))
              #  model.add(LSTM(UNITS, return_sequences=False, stateful=False))

                model.add(Dropout(DROPOUT_RATE))
                model.add(Dense(NUM_CLASSES, activation='softmax'))

                model.compile(loss=loss(weights),  # weighted_categorical_crossentropy(weights),
                              optimizer=tf.keras.optimizers.Adam(
                                  learning_rate=LEARNING_RATE),
                              metrics=[keras.metrics.categorical_accuracy])
                return model
        elif method == 'lstm-fcn':
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
            def create_model():
                ip = Input(shape=(X.shape[1],X.shape[2]))

                x = Masking()(ip)
                x = LSTM(UNITS)(x)
                
                #x = LSTM(8,return_sequences=True)(x)
                x = Dropout(DROPOUT_RATE)(x)
               # x = Dense(128)(x)  # Adding a dense layer to match the output shape of the LSTM

               # y = Permute((2, 1))(ip)
                y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = squeeze_excite_block(y)
                print(y.shape)

                y = Conv1D(256, 5, padding='causal', kernel_initializer='he_uniform')(y)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = squeeze_excite_block(y)
                print(y.shape)

                y = Conv1D(128, 3, padding='causal', kernel_initializer='he_uniform')(y)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                print(y.shape)

                y = GlobalAveragePooling1D()(y)

                #y = Flatten()(y)
    #            print(y.shape)



                x = concatenate([x, y])
                

                out = Dense(NUM_CLASSES, activation='softmax')(x)

                model = Model(ip, out)
               # model.summary()
                model.compile(loss=loss(weights),
                             optimizer=tf.keras.optimizers.Adam(
                                 learning_rate=LEARNING_RATE),
                             metrics=[keras.metrics.categorical_accuracy])
                # add load model code here to fine-tune

                return model
        tensorboard_callback = TensorBoard(
            log_dir=folder_path_tensorboard, histogram_freq=1)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=max(
            2, np.round(EPOCHS/10)), restore_best_weights=True, min_delta=0.001)
        lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, np.round(EPOCHS/20)))
        
        model_callbacks = [tensorboard_callback, early_stopping, lr_reduction]
    model=create_model()
    return model,model_callbacks
