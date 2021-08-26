"""
Red neuronal con Tensor Flow
Convertir grados Celsius a Fahrenheit
F = C.1,8+32
Tambien se usa el framework Keras, ya que facilita mucho el
codigo

Una capa de entrada y otra de salida
"""
# importo librerias
import tensorflow as tf
import numpy as np
import seaborn as sns

# red con una capa de entrada y otra de salida

# declaro entradas y salidas
celsius = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100],dtype=float)

# units=cantidad de neuronas de la capa
# input_shape, con eso le decimos que tenemos una entrada
# con una neurona, se autoregistra la capa con una neurona
# capa = tf.keras.layers.Dense(units=1,input_shape=[1])

# para este caso uso dos capas ocultas y una de salida
oculta1 = tf.keras.layers.Dense(units=3,input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
 
# necesito usar un modelo de keras para trabajar con las capas
modelo = tf.keras.Sequential([oculta1,oculta2,salida])

# compilo el modelo
# el optimizador Adam permite a la red ajustar los sesgos
# de manera eficiente, lo que evita el sobreajuste o que 
# desaprenda
# loss, funcion de perdidas
# mean_squared_error considera que pocos errores grandes
# son peores que muchos errores pequenos
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
    )
# ahora se entrena modelo
print('Se comienza entrenamiento')
historial = modelo.fit(celsius,fahrenheit,
                       epochs=100,
                       verbose=False)
print('Modelo entrenado!')

# luego de entrenado se quiere conocer los resultados de la 
# funcion de perdidas, que nos dice que tan mal estan los 
# resultados de la red en cada epoch
import matplotlib.pyplot as plt
# plt.xlabel('# Epoch')
# plt.ylabel('Magnitud de perdida')
# plt.grid()
# plt.plot(historial.history['loss'])
# plt.savefig('02-funcion de perdidas red neuronal')
# como se observa en el grafico, despues de las 400epochs
# los errores se mantienen casi constantes, por lo tanto
# se podria ajustar la cantidad de epochs

# prueba de la red neuronal
print('Se realiza prediccion de prueba')
celsius_prueba = np.array([-32,-6,-1,20,28,45,52,78,90,110,111,116,150,174,180,199,234],dtype=float)
fahrenheit_calculo = lambda x: x*1.8+32
fahrenheit_prueba = []
simulador = []
# calculados directamente por la formula
for C in celsius_prueba:
    fahrenheit_prueba.append(fahrenheit_calculo(C))
    simulador.append(modelo.predict([C]))
    
# grafico los datos calculados por la red vs los datos
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(celsius_prueba, simulador, marker="x", label='prediccion')
ax1.scatter(celsius_prueba,fahrenheit_prueba, marker="*", label='conversion directa')
plt.legend(loc='upper left');
plt.xlabel('Grados Celsius')
plt.xlabel('Grados Fahrenheit')
plt.grid()
plt.savefig('02-comparacion prediccion vs calculo')
plt.show()


# se quieren conocer los valores se sesgo y bias asignados 
# por la red
print('Variables internas del modelo')
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())

