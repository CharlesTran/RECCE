import pandas as pd

data = pd.read_csv('/data/czx/dataset/phase1/trainset_label.txt')
nums = data['target'].value_counts()
print(len(nums))
print(nums)