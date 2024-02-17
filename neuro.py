from tensorflow import keras
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
import pandas, numpy



text = pandas.read_csv('train.csv')

print(text.columns)
# print(text.describe())

# Preparing of data
price = numpy.array(text['SalePrice'].tolist(), dtype='float64')

year_sold = numpy.array(text['YrSold'].tolist(), dtype='float64')
area = numpy.array(text['LotArea'].tolist(),  dtype='float64')
year_built = numpy.array(text['YearBuilt'].tolist(),  dtype='float64')
overall_q = numpy.array(text['OverallQual'].tolist(),  dtype='float64')
overall_c = numpy.array(text['OverallCond'].tolist(),  dtype='float64')
# print(text.head(5))
# print(text['SalePrice'].tolist())

# Preprocessing of data
year_sold_min = min(year_sold)
year_sold -= year_sold_min
year_sold_max = max(year_sold)
year_sold /= max(year_sold)

year_built_min = min(year_built)
year_built -= year_built_min
year_built_max = max(year_built)
year_built /= max(year_built)

area_min = min(area)
area -= area_min
area_max = max(area)
area /= max(area)

price_min = min(price)
price -= price_min
price_max = max(price)
price /= max(price)

overall_q_min = min(overall_q )
overall_q -= overall_q_min
overall_q_max = max(overall_q )
overall_q /= max(overall_q )

overall_c_min = min(overall_c)
overall_c -= overall_c_min
overall_c_max = max(overall_c)
overall_c /= max(overall_c)

# training prepare
y_train = price.reshape(len(price), 1)

area = area.reshape(len(area), 1)
year = year_sold.reshape(len(year_sold), 1)
built = year_built.reshape(len(year_built), 1)
qual = overall_q.reshape(len(year_built), 1)
cond = overall_c.reshape(len(year_built), 1)

x_train = numpy.hstack([area, year, built, qual, cond])
print(x_train)

f = open('config.txt', 'w')
f.write(str(year_sold_min)+'\n')
f.write(str(year_sold_max)+'\n')
f.write(str(year_built_min)+'\n')
f.write(str(year_built_max)+'\n')
f.write(str(area_min)+'\n')
f.write(str(area_max)+'\n')
f.write(str(price_min)+'\n')
f.write(str(price_max)+'\n')
f.write(str(overall_q_min)+'\n')
f.write(str(overall_q_max)+'\n')
f.write(str(overall_c_min)+'\n')
f.write(str(overall_c_max)+'\n')
f.close()


#model

model = Sequential()
model.add(Input(5))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="relu"))
print(model.summary())

model.compile(loss='mse', metrics=['mse'])

# Сохранение модели
# model.save()
model.fit(x_train, y_train, epochs= 150, batch_size=32)
# # for i in range(20):
#     model.fit(x_train, y_train, epochs= 25, batch_size=32)
#     model.save(f"models/model {i}")
model.evaluate(x_train)
model.save('models/model_main')
# Загрузка
# model = model.load(file_name)


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
# ***




'''Выборки:
 тренировочная
 валидационная - график ошибки от эпох => выборка, которой не было в тренировочной для прогона = > ошибка побольше и скачок - границы переобучения
 Методы остановки обучения для исключения переобучения 
 тестовая - преобразуем в препроцессинг, вставляем в эвалюэйт => получает ошибку для отладки
 
 слои бывают полносвязными
 сверточными - обработка изображений
 рекурентные 
 '''
