import tensorflow as tf
import parameters as params


class bidirectional_model(tf.keras.Model):
    def __init__(self, num_bidirectional_layers=params.NUM_GRU_LAYERS):
        super(bidirectional_model, self).__init__()
        self.bidirectional_layers = [
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(params.GRU_WIDTH, return_sequences=True))
                                     for _ in range(num_bidirectional_layers)
                                    ]
        
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, data_aug=True):
        
        for layer in self.bidirectional_layers:
            x = layer(x)
        return tf.reshape(self.final_layer(x), [self.final_layer(x).shape[0], self.final_layer(x).shape[1]])
    
class bidirectional_model_for_save(tf.keras.Model):
    def __init__(self, num_bidirectional_layers=params.NUM_GRU_LAYERS):
        super(bidirectional_model_for_save, self).__init__()
        self.bidirectional_layers = [
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(params.GRU_WIDTH, return_sequences=True))
                                     for _ in range(num_bidirectional_layers)
                                    ]
        
        self.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, data_aug=True):
        
        for layer in self.bidirectional_layers:
            x = layer(x)
        return self.final_layer(x)

def save_weights(model, model_save, name_saving_folder):
    """
    To save the weights of the model. As in the last layer there is a reshape, doing model.save() directly
    gives an error.
    
    In order for save_weights not to give an error, model_save has to be used at least once, to set the sizes of its weights.
    """
    model_save.set_weights(model.get_weights())
    model_save.save(name_saving_folder)
