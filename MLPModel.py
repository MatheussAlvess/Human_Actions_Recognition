import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class MLP():
    def __init__(self,
                dataset_path: object = 'dataset/coords.csv',
                epochs: int = 50,
                batch_size: int = 32,
                test_size: float = 0.3,
                validation_split: float = 0.2,
                ff_activation: object = 'softmax',
                optimizer: object = 'adam',
                loss: object ='sparse_categorical_crossentropy',
                save: bool = True) -> None:
        
        self.dataset_path = dataset_path
        self.test_size = test_size
        self.ff_activation = ff_activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.optimizer = optimizer
        self.loss = loss
        self.save = save


    def prepare_dataset(self,dataset_path):

        df = pd.read_csv(dataset_path)
        df['label'] = df['label'].apply(lambda x: x.replace('1',''))

        X = df.drop('label', axis=1) # features
        y = df['label'] # target value

        # Transformar a coluna 'label' em valores numéricos
        self.class_map = {class_label: idx for idx, class_label in enumerate(df['label'].unique())}
        df['label'] = df['label'].map(self.class_map)

        # Dividir os dados em features (X) e rótulos (y)
        X = df.drop(columns=['label'])
        y = df['label']
        
        # Dividir os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        return X_train, X_test, y_train, y_test

    def run_mlp(self):

        X_train, X_test, y_train, y_test = self.prepare_dataset(self.dataset_path)

        # Construir o modelo MLP
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.5),  # Adiciona uma camada de dropout para regularização
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),  # Adiciona outra camada de dropout
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(self.class_map), activation=self.ff_activation)  # Saída com softmax para classificação multiclasse
        ])

        # Compilar o modelo
        self.model.compile(optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=['accuracy'])

        # Treinar o modelo
        self.model.fit(X_train.values, y_train.values, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)

        # Avaliar o modelo no conjunto de teste
        self.test_loss, self.test_accuracy = self.model.evaluate(X_test, y_test)

        if self.save:
            self.model.save('mlp_model.h5')


if __name__=='__main__':
    mlp = MLP(epochs=100,validation_split=0.1)
    mlp.run_mlp()

    print(f'Loss no conjunto de teste: {mlp.test_loss}')
    print(f'Acurácia no conjunto de teste: {mlp.test_accuracy}')
