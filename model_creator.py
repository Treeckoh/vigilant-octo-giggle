import tensorflow as tf
from tensorflow.keras import layers, Sequential


class ModelCreator:
    def __init__(
        self,
        X,  # Pandas DF
        y, 
        val_X,
        val_y,
        model_name,
        loss = 'mae'
    ):
        self.model_name = model_name
        self.X = X
        self.y = y
        self.val_X = val_X
        self.val_y = val_y
        self.input_size = len(X.columns)
        self.output_size = len(y.columns)
        self.loss = loss
        self.run()
    
    def create_model(self):
        inputs = layers.Input(shape=(self.input_size))
        
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        outputs = layers.Dense(self.output_size, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(loss=self.loss,
                           optimizer='Adam',
                           metrics=['mae'])
        
    def fit_model(self):
        self.model.fit(
            self.X,
            self.y,
            validation_data=(self.val_X, self.val_y),
            epochs=100,
            callbacks=tf.keras.callbacks.EarlyStopping(
                patience=10,
                verbose=0,
                restore_best_weights=True
            )
        )
        self.model.save(f'{self.model_name}.keras')
        
    def run(self):
        self.create_model()
        self.fit_model()
        