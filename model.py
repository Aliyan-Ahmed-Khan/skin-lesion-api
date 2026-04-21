import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)

def cbam_block(feature_map, ratio=8):

    channel = feature_map.shape[-1]

    shared_dense_one = layers.Dense(channel // ratio, activation='relu')
    shared_dense_two = layers.Dense(channel)

    avg_pool = layers.GlobalAveragePooling2D()(feature_map)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(feature_map)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    channel_attention = layers.Add()([avg_pool, max_pool])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    channel_attention = layers.Reshape((1,1,channel))(channel_attention)

    channel_refined = layers.Multiply()(
        [feature_map, channel_attention]
    )

    avg_pool_spatial = layers.Lambda(
        lambda x: tf.reduce_mean(
            x,
            axis=-1,
            keepdims=True
        )
    )(channel_refined)

    max_pool_spatial = layers.Lambda(
        lambda x: tf.reduce_max(
            x,
            axis=-1,
            keepdims=True
        )
    )(channel_refined)

    concat = layers.Concatenate(axis=-1)(
        [avg_pool_spatial, max_pool_spatial]
    )

    spatial_attention = layers.Conv2D(
        1,
        kernel_size=7,
        padding='same',
        activation='sigmoid'
    )(concat)

    return layers.Multiply()(
        [channel_refined, spatial_attention]
    )

def build_model(num_classes):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224,224,3),
        include_top=False,
        weights=None
    )

    x = cbam_block(base_model.output)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    return models.Model(
        inputs=base_model.input,
        outputs=output
    )
