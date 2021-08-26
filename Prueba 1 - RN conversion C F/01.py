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

# red con una capa de entrada y otra de salida

# declaro entradas y salidas
celsius = np.array([-40,-10,0,8,15,22,38],dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100],dtype=float)

# units=cantidad de neuronas de la capa
# input_shape, con eso le decimos que tenemos una entrada
# con una neurona, se autoregistra la capa con una neurona
capa = tf.keras.layers.Dense(units=1,input_shape=[1])
# necesito usar un modelo de keras para trabajar con las capas
modelo = tf.keras.Sequential([capa])

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
                       epochs=1000,
                       verbose=False)
print('Modelo entrenado!')

# luego de entrenado se quiere conocer los resultados de la 
# funcion de perdidas, que nos dice que tan mal estan los 
# resultados de la red en cada epoch
import matplotlib.pyplot as plt
plt.xlabel('# Epoch')
plt.ylabel('Magnitud de perdida')
plt.grid()
plt.plot(historial.history['loss'])
plt.savefig('01-funcion de perdidas red neuronal')
# como se observa en el grafico, despues de las 400epochs
# los errores se mantienen casi constantes, por lo tanto
# se podria ajustar la cantidad de epochs

# prueba de la red neuronal
print('Se realiza prediccion de prueba')
resultado = modelo.predict([100.0])
print('Resultado de la prediccion: '+str(resultado[0][0])+'F')

# se quieren conocer los valores se sesgo y bias asignados 
# por la red
print('Variables internas del modelo')
print(capa.get_weights())

