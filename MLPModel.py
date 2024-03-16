import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split


class MLP():
    def __init__(self,
                dataset_path: object = 'dataset/coords.csv',  # Caminho do conjunto de dados 
                epochs: int = 50,  # Numero de épocas de treinamento 
                batch_size: int = 32,  # Tamanho do batch 
                test_size: float = 0.3,  # Tamanho do conjunto de teste 
                validation_split: float = 0.2,  # Proporção de divisão do conjunto de validação 
                ff_activation: object = 'softmax',  # Função de ativação da camada de saída 
                optimizer: object = 'adam',  # Algoritmo de otimização 
                loss: object ='sparse_categorical_crossentropy',  # Função de perda 
                save: bool = True) -> None:  # Flag para salvar o modelo 
        
        # Inicializa os parametros da rede neural
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
        # Carrega o conjunto de dados a partir do caminho especificado
        df = pd.read_csv(dataset_path)
        df['label'] = df['label'].apply(lambda x: x.replace('1',''))

        # Remove o '1' dos rótulos
        X = df.drop('label', axis=1) # features
        y = df['label'] # target value

        # Mapeia os rótulos para valores numéricoss
        self.class_map = {class_label: idx for idx, class_label in enumerate(df['label'].unique())}
        df['label'] = df['label'].map(self.class_map)

        # Dividir os dados em features (X) e labels (y)
        X = df.drop(columns=['label'])
        y = df['label']
        
        # Dividir os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        return X_train, X_test, y_train, y_test

    def run_mlp(self):
        # Prepara o conjunto de dados
        X_train, X_test, y_train, y_test = self.prepare_dataset(self.dataset_path)

        # Construção do modelo MLP
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)), # Especifica a forma de entrada
            tf.keras.layers.Dropout(0.5),  # Adicionando uma camada de dropout
            tf.keras.layers.Dense(64, activation='relu'), # Adiciona camada oculta
            tf.keras.layers.Dropout(0.5),  # Adiciona outra camada de dropout
            tf.keras.layers.Dense(32, activation='relu'), # Adiciona outra camada oculta 
            tf.keras.layers.Dense(len(self.class_map), activation=self.ff_activation)  # Saída com softmax para classificação multiclasse
        ])

        # Compila o modelo
        self.model.compile(optimizer=self.optimizer,
                    loss=self.loss,
                    metrics=['accuracy'])

        # Treina o modelo
        self.model.fit(X_train.values, y_train.values, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split)

        # Avalia o modelo no conjunto de teste
        self.test_loss, self.test_accuracy = self.model.evaluate(X_test, y_test)

        if self.save:
            self.model.save('mlp_model.h5')


if __name__=='__main__':
    mlp = MLP(epochs=100,validation_split=0.1)
    mlp.run_mlp()

    print(f'Loss no conjunto de teste: {mlp.test_loss}')
    print(f'Acurácia no conjunto de teste: {mlp.test_accuracy}')