from sklearn.model_selection import train_test_split


class DuplicateDataSet:
    """sets the same training and testing dataset and returns list with tuple the given number of times [(x_train, x_test, y_train, y_test)] 

       Args:
        dataset (DataFrame): dataset with the info 
        amount (int): amount of sets return
        column_X (str): column where to extract the information obtained from X 
        column_y (str): column where to extract the information obtained from Y
       Return (list): 
    """

    def __init__(self, dataset, amount, column_X, column_y) -> list:
        self.dataset = dataset
        self.amount = amount
        self.column_X = column_X
        self.column_y = column_y
        self._set_same_train_and_test()

    def _set_same_train_and_test(self):
        X = self.dataset[self.column_X]
        y = self.dataset[self.column_y]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, train_size=0.8)

        list_of_tuples = []
        for i in range(self.amount):
            list_of_tuples.append((X_train, X_test, y_train, y_test))

        return list_of_tuples


def set_same_train_and_test(dataset, column_X, column_y):
    X = dataset[column_X]
    y = dataset[column_y]

    return train_test_split(X, y, test_size=0.2, train_size=0.8)