from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.keras.layers import Dense

# Trains a model and saves it to disk
def train():
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

    X = dataset[:, 0:8]
    y = dataset[:, 8]

    model = Sequential([
        Dense(12, input_dim=8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')]
    )

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=150, batch_size=10, verbose=0)

    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))

    model.summary()
    model.save('model.tf', save_format='tf')

# Loads a model from disk
def load():
    return load_model('model.tf')

# Make a prediction using a model
def predict(model, X):
    return [round(y.tolist()[0]) == 1 for y in model.predict([X])][0]

# Check that everything works as expected
def test():
    m = load()

    predict(m, [1,85,66,29,0,26.6,0.351,31])
    predict(m, [8,183,64,0,0,23.3,0.672,32])
    predict(m, [1,89,66,23,94,28.1,0.167,21])
    predict(m, [0,137,40,35,168,43.1,2.288,33])
    predict(m, [5,116,74,0,0,25.6,0.201,30])
    predict(m, [3,78,50,32,88,31.0,0.248,26])
    predict(m, [10,115,0,0,0,35.3,0.134,29])
    predict(m, [2,197,70,45,543,30.5,0.158,53])
    predict(m, [8,125,96,0,0,0.0,0.232,54])
    predict(m, [4,110,92,0,0,37.6,0.191,30])
    predict(m, [10,168,74,0,0,38.0,0.537,34])
    predict(m, [10,139,80,0,0,27.1,1.441,57])
    predict(m, [1,189,60,23,846,30.1,0.398,59])
    predict(m, [5,166,72,19,175,25.8,0.587,51])
    predict(m, [7,100,0,0,0,30.0,0.484,32])
    predict(m, [0,118,84,47,230,45.8,0.551,31])
    predict(m, [7,107,74,0,0,29.6,0.254,31])

# To train module, run this script as `python -m dnn`
if __name__ == "__main__":
    train()
