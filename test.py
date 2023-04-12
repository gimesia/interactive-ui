import pandas as pd
import numpy as np

df = pd.DataFrame({'name': ['A', 'B', 'C'], 'center_point': [(1, 2), (2, 3), (3, 4)]})
print(df)

# convert the tuples to separate x and y columns
df['x'], df['y'] = zip(*df['center_point'])
print(df)

# calculate the euclidean distance from point (5,5)
df['euclidean_distance'] = np.sqrt((df['x'] - 5) ** 2 + (df['y'] - 5) ** 2)
print(df)
