#%%
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, Dropout, Flatten, Dense, Activation, Layer, Reshape, Permute, Lambda
from tensorflow.keras.layers import Conv3D, MaxPool3D, ZeroPadding3D
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras import backend as K
from tensorflow.summary import scalar

def UNet(shape, 
         nClasses=1, 
         loss="binary_crossentropy", 
         lr=1e-5, 
         metrics=['accuracy']):
    
    IMG_HEIGHT = shape[0]
    IMG_WIDTH = shape[1]
    IMG_CHANNELS = nClasses

    # Build U-Net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2D(256, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    
    up8 = Conv2D(128, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1, up9])

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    output_shape = Model(inputs , conv9 ).output_shape

    output = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs, output)
    model.compile(loss=loss, optimizer = Adam(lr = lr) , metrics=metrics)
    model.summary()

    return model

def UNet_hp(shape, 
            hparams,
            hparams_list, 
            nClasses=1, 
            loss="binary_crossentropy", 
            lr=1e-5, 
            metrics=['accuracy'], ):
    
    IMG_HEIGHT = shape[0]
    IMG_WIDTH = shape[1]
    IMG_CHANNELS = nClasses
    
    HP_DROPOUT = hparams_list[0]
    
    # Build U-Net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    drop4 = Dropout(hparams[HP_DROPOUT])(conv4)
    
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    drop5 = Dropout(hparams[HP_DROPOUT])(conv5)
    
    up6 = Conv2D(512, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    merge6 = concatenate([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(256, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2D(128, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, 
                 activation = 'relu', 
                 padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    
    merge9 = concatenate([conv1, up9])

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    output_shape = Model(inputs , conv9 ).output_shape

    output = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs, output)
    
    model.compile(loss=loss, optimizer = Adam(lr = lr) , metrics=metrics)
    
    model.summary()

    return model 

# ---------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------

def conv2d_block(
    inputs, 
    use_batch_norm = True, 
    dropout=0.3, 
    filters=16, 
    kernel_size=(3,3), 
    activation='relu', 
    kernel_initializer='he_normal', 
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(inputs)
    # the kernel is relu, we can skip "scale step" in batch normalization
    # x = BatchNormalization(axis=-1, center=True, scale=False)(x)
    if use_batch_norm:
        c = BatchNormalization(axis=-1, center=True, scale=False)(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    c = Conv2D(filters, kernel_size, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(c)
    if use_batch_norm:
        c = BatchNormalization(axis=-1, center=True, scale=False)(c)
    return c

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

# ---------------------------------------------------------------------

def vanilla_unet(
    shape,
    dropout = 0.5, 
    num_layers = 4,
    num_classes = 1,
    filters = 64,
    output_activation = 'sigmoid', # 'sigmoid' or 'softmax'
    loss = "binary_crossentropy", 
    lr = 1e-5, 
    metrics = ['accuracy'], 
    summary = True,
    use_batch_norm = False,
    ): 
    
    
    # Build U-Net model
    inputs = Input((shape[0], shape[1], num_classes))
    x = inputs 
    
    # Encoder
    down_layers = []
    for l in range(num_layers):
        
        # add conv block
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, padding='same')
        down_layers.append(x)
        # MaxPooling
        x = MaxPool2D((2, 2))(x)
        filters = filters*2 # double the number of filters with each layer
        
    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, padding='same')
    
    # Decoder
    for conv in reversed(down_layers):
        
        filters //= 2 # decreasing number of filters with each layer     
        # UpSampling
        x = UpSampling2D(size = (2,2))(x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, padding='same')
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # compile model
    model.compile(loss=loss, optimizer = Adam(lr = lr) , metrics=metrics)
    
    # print summary
    if summary :
        model.summary()
    
    return model

def vanilla_unet_nodrop(
    shape,
    dropout = 0.5, 
    num_layers = 4,
    num_classes = 1,
    filters = 64,
    output_activation = 'sigmoid', # 'sigmoid' or 'softmax'
    loss = "binary_crossentropy", 
    lr = 1e-5, 
    metrics = ['accuracy'], 
    summary = True,
    use_batch_norm = False,
    ): 
    
    # Build U-Net model
    inputs = Input((shape[0], shape[1], num_classes))
    x = inputs 
    
    # Encoder
    down_layers = []
    for l in range(num_layers):
        
        # add conv block
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=0, padding='same')
        down_layers.append(x)
        # MaxPooling
        x = MaxPool2D((2, 2))(x)
        filters = filters*2 # double the number of filters with each layer
        
    x = Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, padding='same')
    
    # Decoder
    for conv in reversed(down_layers):
        
        filters //= 2 # decreasing number of filters with each layer     
        # UpSampling
        x = UpSampling2D(size = (2,2))(x)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=0, padding='same')
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation)(x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    # compile model
    model.compile(loss=loss, optimizer = Adam(lr = lr) , metrics=metrics)
    
    # print summary
    if summary :
        model.summary()
    
    return model
