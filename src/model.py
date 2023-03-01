import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#  from src import settings


class DecisionTree():
    def __init__(self, dataset):
        self.dataset = pd.read_csv(dataset)
        self.tree_model = DecisionTreeClassifier(criterion='entropy',
                                                 max_depth=10,
                                                 min_samples_split=2,
                                                 random_state=42)

    def split_dataset(self):
        inp = self.dataset.drop(['output'], axis=1).values
        out = self.dataset['output'].values
        # pylint: disable=unused-variable
        x_train, x_test, y_train, y_test = train_test_split(inp, out, test_size=0.2, random_state=42)

        return x_train, y_train

    def fit_model(self):
        x_train, y_train = self.split_dataset()
        self.tree_model.fit(x_train, y_train)

    def dump_model(self, filename):
        joblib.dump(self.tree_model, f'data/{filename}.sav')

    def predict(self, to_predict):
        return self.tree_model.predict(to_predict)


tree = DecisionTree('data/heart.csv')
tree.fit_model()
tree.dump_model('tree_model')
# loaded_model = pickle.load(open('data/tree_model.sav', 'rb'))
# loaded_model.predict([[35, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
