def predict():
    import numpy as np
    import pandas as pd 
    import tensorflow as tf
    samplexl=pd.read_csv("통합 문서1.csv", header=0)
    sample_data=samplexl[['radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean']].values
    model=tf.keras.models.load_model('projectmodel.h5')
    if np.argmax(model.predict(sample_data))==1:
        print('malignant tumor')
    else:
        print('benign tumor')
def main():
    import csv
    radius_mean=float(input('Enter radius mean\n'))
    perimeter_mean=float(input('Enter perimeter mean\n'))
    area_mean=float(input('Enter area_mean\n'))
    compactness_mean=float(input('Enter compactness mean\n'))
    concavity_mean=float(input('Enter concavity mean\n'))
    concave_points_mean=float(input('Enter concave points mean\n'))
    f=open('통합 문서1.csv','w', newline='')
    wr=csv.writer(f)
    wr.writerow(['id', 'radius_mean','perimeter_mean','area_mean','compactness_mean','concavity_mean','concave points_mean'])
    wr.writerow([1,radius_mean,perimeter_mean, area_mean, compactness_mean, concavity_mean, concave_points_mean])
    f.close()
    predict()
    input('Press enter to exit')
if __name__ == '__main__':
    main()