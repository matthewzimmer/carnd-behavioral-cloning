from basic_network import train_network

train_network(
    nb_epoch=2,
    batch_size=32,
    validation_split_percentage=0.05,
    output_shape=(40, 80, 3),
    learning_rate=0.001,
    dropout_prob=0.1,
    activation='relu'
)