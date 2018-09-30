from tensorflow import keras

def word_level_cnn(num_words, num_features):
    model = keras.Sequential()
    
    # input
    input = keras.layers.InputLayer(input_shape=[num_words, num_features])
    model.add(input)

    kernels = [3, 5, 7, 11, 11, 11] #[11, 7, 5, 3, 3, 3]
    filters = [32, 64, 64, 64, 64, 64] #[8, 16, 32, 32, 32, 32]
    has_norm = [True, True, True, False, False, False] #[True, True, False, False, False, False]
    has_pool = [True, True, True, False, False, False] #[True, True, False, False, False, False]
    dense_units = [512, 512] #[1024, 1024]

    # conv
    for i in range(len(kernels)):
        conv = keras.layers.Conv1D(filters[i], kernels[i], activation='relu', padding='same')
        model.add(conv)

        if has_norm[i]:
            norm = keras.layers.BatchNormalization()
            model.add(norm)
        
        if has_pool[i]:
            pool = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='valid')
            model.add(pool)

    model.add(keras.layers.Flatten())

    # dense
    for unit in dense_units:
        dense = keras.layers.Dense(unit, 'relu')
        model.add(dense)

        dropout = keras.layers.Dropout(0.5)
        model.add(dropout)

    output = keras.layers.Dense(2, activation='sigmoid')
    model.add(output)

    model.compile(optimizer='adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()
    return model

def char_level_cnn(num_chars, vocab_size):
    model = keras.Sequential()

    # embed
    embed = keras.layers.Embedding(vocab_size, vocab_size, input_length=num_chars)
    model.add(embed)

    kernels = [7, 7, 3, 3, 3, 3]
    filters = [256, 256, 256, 256, 256, 256]
    has_norm = [True, True, False, False, False, True]
    has_pool = [True, True, False, False, False, True]
    dense_units = [1024, 1024, 1024]

    for i in range(len(kernels)):
        conv = keras.layers.Conv1D(filters[i], kernels[i], activation='relu', padding='valid')
        model.add(conv)

        if has_norm[i]:
            norm = keras.layers.BatchNormalization()
            model.add(norm)
        
        if has_pool[i]:
            pool = keras.layers.MaxPool1D(pool_size=3, padding='valid')
            model.add(pool)

    model.add(keras.layers.Flatten())

    for i in range(len(dense_units)):
        unit = dense_units[i]
        dense = keras.layers.Dense(unit, 'relu')
        model.add(dense)

        is_last_layer = (i == len(dense_units) - 1)
        if not is_last_layer:
            dropout = keras.layers.Dropout(0.5)
            model.add(dropout)

    output = keras.layers.Dense(2, activation='sigmoid')
    model.add(output)

    model.compile(optimizer= 'adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()
    return model