import scipy.io as sio
import numpy as np
from SAR_utils import *

#from tensorflow import keras
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from Load_Data import load_data

#from tensorflow.keras.utils import plot_model
from cvnn.layers import complex_input, ComplexConv2D, ComplexConv3D, ComplexDense, ComplexDropout, ComplexFlatten
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from net_flops import net_flops

from CoordAttention import CoordAtt_cmplx

# Get the data
dataset = 'FL_T'
windowSize = 15
test_ratio = 0.99
data, gt = load_data(dataset)
if dataset =='ober':
    lr = 0.0001
else:
    lr = 0.001
    

data = Standardize_data(data)


X_coh, y = createImageCubes(data, gt, windowSize)
X_coh = np.expand_dims(X_coh, axis=4)


X_train, X_test, y_train, y_test = splitTrainTestSet(X_coh, y, test_ratio)
del X_coh # To save RAM memory


total = 0
numm = []
for i in range(int(np.max(y_test)+1)):
    tmp = np.sum(y_test==i)
    total = total + tmp
    numm.append(tmp)
    print("Class #"+str(i) +": " + str(tmp))
   
    
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


image_size = windowSize  # Final Image Size
patch_size = 3  # Patch Dimension
num_patches = (image_size // patch_size) ** 2
projection_dim = 32
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4 #8
mlp_head_units = [1024, 512] #[2048, 1024]  # Size of the dense layers


"""## Implementing Multilayer Perceptron"""
def cmplx_multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = ComplexDense(units, activation=cart_gelu)(x)
        x = ComplexDropout(dropout_rate)(x)
    return x

"""## Implementing patch creation as a layer"""
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


"""## Implement the Patch Encoding Layer"""
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = ComplexDense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + tf.cast(self.position_embedding(positions), tf.complex64)
        return encoded


def MultiScaleFeatureExtractor(inputs):    
    # Spatial Path
    x1 = ComplexConv3D(filters=8, kernel_size=(3, 3, 1), activation='cart_relu', padding='same')(inputs)
    x1 = ComplexConv3D(filters=8, kernel_size=(3, 3, 1), activation='cart_relu',padding='same')(x1)
    
    # Polarimtric Path
    x2 = ComplexConv3D(filters=8, kernel_size=(1, 1, 3), activation='cart_relu',padding='same')(inputs)
    x2 = ComplexConv3D(filters=8, kernel_size=(1, 1, 3), activation='cart_relu',padding='same')(x2)

    # Spatial-Polarimtric Path
    x3 = ComplexConv3D(filters=8, kernel_size=(3, 3, 3), activation='cart_relu',padding='same')(inputs)
    x3 = ComplexConv3D(filters=8, kernel_size=(3, 3, 3), activation='cart_relu',padding='same')(x3)

    concatenated_features = tf.concat([x1,x2,x3],axis=4);
    
    return concatenated_features


def cmplx_ViT(x):
   
    patches = Patches(patch_size)(x)

    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)


    for _ in range(transformer_layers):

        x1_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(encoded_patches))
        x1_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(encoded_patches))
        
        attention_output_r = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1_r, x1_r)
        attention_output_i = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1_i, x1_i)
        attention_output = tf.complex(attention_output_r, attention_output_i)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(x2))
        x3_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(x2))
        x3 = tf.complex(x3_r, x3_i)
        
        
        x3 = cmplx_multilayer_perceptron(x3, hidden_units=transformer_units, dropout_rate=0.1)

        encoded_patches = layers.Add()([x3, x2])

    representation_r = layers.LayerNormalization(epsilon=1e-6)(tf.math.real(encoded_patches))
    representation_i = layers.LayerNormalization(epsilon=1e-6)(tf.math.imag(encoded_patches))
    representation = tf.complex(representation_r, representation_i)
    
    representation = ComplexFlatten()(representation)
    representation = ComplexDropout(0.5)(representation)

    features = cmplx_multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    
    return features

def MyModel(img_list, num_class):
    inputs = complex_input(shape=img_list.shape[1:])
    
    x = MultiScaleFeatureExtractor(inputs)
    x_shape = x.shape
    x = keras.layers.Reshape((x_shape[1], x_shape[2], x_shape[3]*x_shape[4]))(x)
    x = ComplexConv2D(filters=24, kernel_size=(3,3), activation='cart_relu',padding='same')(x)
   
    
    x = CoordAtt_cmplx(x, 4)
    
    
    x = cmplx_ViT(x)
    
    z = ComplexFlatten()(x)
    logits = ComplexDense(num_classes(dataset), activation="softmax_real_with_abs")(z)
    
    model = tf.keras.Model(inputs=[inputs], outputs=logits)
    model.compile(optimizer = Adam(learning_rate=lr), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    return model

"""## Compile, Train, and Evaluate the model"""
model = MyModel(X_train, num_classes(dataset))
model.summary()

net_flops(model)
# Perform Training
from tensorflow.keras.callbacks import EarlyStopping
early_stopper = EarlyStopping(monitor='accuracy', 
                              patience=10,
                              restore_best_weights=True
                              )

model = MyModel(X_train, num_classes(dataset))
    
history = model.fit(X_train, y_train,
                            batch_size = 128, 
                            verbose = 1, 
                            epochs = 100, 
                            shuffle = True,
                            callbacks = [early_stopper] )
    


Y_pred_test = model.predict([X_test])
y_pred_test = np.argmax(Y_pred_test, axis=1)
       
    
    
    
kappa = cohen_kappa_score(np.argmax(y_test, axis=1),  y_pred_test)
oa = accuracy_score(np.argmax(y_test, axis=1), y_pred_test)
confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_test)
each_acc, aa = AA_andEachClassAccuracy(confusion)
    

    
 

print("oa = ", format((oa)*100, ".2f")) 
print("aa = ", format((aa)*100, ".2f"))
print('Kappa = ', format((kappa)*100, ".2f"))


###############################################################################
# Create the predicted class map
del X_train, X_test
X_coh, y = createImageCubes(data, gt, windowSize, removeZeroLabels = False)
X_coh = np.expand_dims(X_coh, axis=4)

Y_pred_test = model.predict(X_coh)
y_pred_test = (np.argmax(Y_pred_test, axis=1)).astype(np.uint8)

Y_pred = np.reshape(y_pred_test, gt.shape) + 1

name = 'CV_MsAtViT_Full'
sio.savemat(name+'.mat', {name: Y_pred})

gt_binary = gt

gt_binary[gt_binary>0]=1


new_map = Y_pred*gt_binary

name = 'CV_MsAtViT'
sio.savemat(name+'.mat', {name: new_map})



