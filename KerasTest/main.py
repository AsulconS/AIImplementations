import numpy as np

from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def main():
    trainSet = read_csv('dataset/train.csv')
    testSet = read_csv('dataset/test.csv')

    # Se descartan las columnas target e id
    Xr = trainSet.drop(['target', 'id'], axis = 1).values
    # Luego, se convierten los datos del target a categóricos
    Yr = to_categorical(trainSet['target'].values)

    # Luego, formateamos los datos:
    Y0r = Yr[Yr[:, 1] == 0] # Filtrando casos negativos
    Y1r = Yr[Yr[:, 1] == 1] # Filtrando casos positivos
    # Luego, concatenamos a partir de los filtros
    rpts = len(Y0r) // len(Y1r)
    Xr = np.concatenate([Xr[Yr[:, 1] == 0], np.repeat(Xr[Yr[:, 1] == 1], rpts, axis = 0)], axis = 0)
    Yr = np.concatenate([Yr[Yr[:, 1] == 0], np.repeat(Yr[Yr[:, 1] == 1], rpts, axis = 0)], axis = 0)

    # Luego, hacemos un split de los datos de entrenamiento y los datos válidos
    Xtra, Xval, Ytra, Yval = train_test_split(Xr, Yr, test_size = 0.3, random_state = 0)

    # Luego, definimos el modelo Secuencial con Keras
    model = Sequential()
    model.add(BatchNormalization(input_shape = tuple([Xtra.shape[1]])))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(BatchNormalization())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(BatchNormalization())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(2, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'RMSprop', metrics = ['accuracy'])

    # Además aplicaremos un Early Stopping para así evitar overfitting (creamos un callback)
    earlyStoppingCallback = EarlyStopping(patience = 10)

    # Creamos un archivo para exportar
    file = open('out.txt', 'w')
    # Exportamos mi nombre
    file.write('Adrian Rolando Bedregal Vento\n\n')
    # Exportamos un summary del modelo (donde se muestra lo solicitado en la rubrica)
    model.summary(print_fn = lambda line: file.write(line + '\n'))

    # Ahora, entrenamos al modelo
    model.fit(Xtra, Ytra, batch_size = 1024, epochs = 200, verbose = 1, callbacks = [earlyStoppingCallback], validation_data = (Xval, Yval))

    # Finalmente testeamos
    Xtst = trainSet.drop(['target', 'id'], axis = 1).values
    Ytst = to_categorical(trainSet['target'].values)
    score, accuracy = model.evaluate(Xtst, Ytst, batch_size = 1024)

    # Y Exportamos los resultados del test
    file.write(f'Test score: {score}\n')
    file.write(f'Test accuracy: {accuracy * 100}%\n')
    file.close()


if __name__ == '__main__':
    main()
