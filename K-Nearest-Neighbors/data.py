import pandas as pd

train = {
    "attr1": [10,4,1,3,4,8,1],
    "attr2": [2,4,9,10,6,8,8],
    "class": ['Yes','No','Yes','Yes','No','No','Yes']
}
test = {
    "attr1": [2,7,1],
    "attr2": [7,7,11]
}
train = pd.DataFrame(train)
test = pd.DataFrame(test)