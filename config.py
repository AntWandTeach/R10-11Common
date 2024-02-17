from tensorflow import keras
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
import pandas, numpy

model = keras.models.load_model('models/model_main')
with open('config.txt') as file:
    year_sold_min = float(file.readline())
    year_sold_max = float(file.readline())
    year_built_min= float(file.readline())
    year_built_max= float(file.readline())
    area_min= float(file.readline())
    area_max= float(file.readline())
    price_min= float(file.readline())
    price_max= float(file.readline())
    overall_q_min= float(file.readline())
    overall_q_max= float(file.readline())
    overall_c_min= float(file.readline())
    overall_c_max= float(file.readline())
# while True:
#     user_year = input('Enter house year sold:')
#     if user_year == 'stop':
#         break
#     else:
#         user_year = ((float(user_year)) - year_sold_min) / year_sold_max
#     user_area = (float(input('Enter house area:')) - area_min) / area_max
#     user_built = (float(input('Enter house year built:')) - year_built_min) / year_built_max
#     user_qual = (float(input('Enter house year built:')) - overall_q_min) / overall_q_max
#     user_cond = (float(input('Enter house year built:')) - overall_c_min) / overall_c_max
#
#     user_dataset = numpy.array([user_area, user_year, user_built, user_qual, user_cond])
#     user_dataset = numpy.expand_dims(user_dataset, axis=0)
#
#     user_price = model.predict(user_dataset)
# # deprocessing
#     user_price = user_price*price_max + price_min
#     print(f"user_price is {user_price}")
g = open('data.txt', 'w')
for i in range(10, 100):
    user_year = ((float(2008)) - year_sold_min) / year_sold_max
    user_area = ((float(45)) - area_min) / area_max
    user_built = (float(1999) - year_built_min) / year_built_max
    user_qual = (float(i/10) - overall_q_min) / overall_q_max
    user_cond = (float(8) - overall_c_min) / overall_c_max

    user_dataset = numpy.array([user_area, user_year, user_built, user_qual, user_cond])
    user_dataset = numpy.expand_dims(user_dataset, axis=0)

    user_price = model.predict(user_dataset)
# deprocessing
    user_price = user_price*price_max + price_min
    g.write(str(i/10) + '   ' + str(user_price[0][0]) + '\n')
