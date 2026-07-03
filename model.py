# model.py
"""
Architecture CNN configurable (nombre de blocs convolutionnels et de couches
denses variable) pour la classification multi-classes de maladies foliaires.
Utilisée avec une recherche aléatoire d'hyperparamètres (voir train.py).
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model(input_shape, num_classes, dropout_rate, l2_val, lr,
                 n_conv_layers=3, n_dense_layers=3):
    """
    Construit un CNN configurable avec nombre de couches variables.
    """
    model = Sequential()
    filters = 64
    for i in range(n_conv_layers):
        model.add(Conv2D(filters, 3, activation='relu', padding='same',
                         kernel_regularizer=l2(l2_val),
                         input_shape=input_shape if i == 0 else None))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2))
        model.add(Dropout(dropout_rate))
        filters *= 2

    model.add(GlobalAveragePooling2D())

    units = 128
    for i in range(n_dense_layers):
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_val)))
        model.add(Dropout(dropout_rate))
        units //= 2

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
