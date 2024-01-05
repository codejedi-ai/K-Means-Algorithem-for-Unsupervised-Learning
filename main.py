import os
import matplotlib.pyplot as plt
from kmeans import MyKmeansApp
# image_name
img_name = 'bunny.bmp'
img = plt.imread(f'images/{img_name}')
app = MyKmeansApp(img, img_name, num_clusters=3, weightXY=1.0, dist_sensitve = 0.3)

app.run()