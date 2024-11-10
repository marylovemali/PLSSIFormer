import numpy as np  
import pandas as pd  
  
# 读取.npy文件  
data = np.load('PEMS04.npz')
print(data['data'].shape)
  
# 将NumPy数组转换为Pandas DataFrame  
df = pd.DataFrame(data['data'])
  
# 将DataFrame保存为.csv文件  
# df.to_csv('PEMS04_test.csv', index=False)
