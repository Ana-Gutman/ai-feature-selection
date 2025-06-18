import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

def separarAtributosObjetivo(nombreArchivocsv, nombreObjetivo):
    datos = pd.read_csv(nombreArchivocsv)
    objetivo= datos[nombreObjetivo]
    atributos = datos.drop(nombreObjetivo, axis=1)
    return atributos, objetivo

def codificacionDeAtributosCategoricos(atributos):
    codificador_atributos_discretos = OrdinalEncoder()
    atributos_categoricos = atributos.select_dtypes(include=['object']).columns.tolist()

    codificador_atributos_discretos.fit(atributos[atributos_categoricos])
    atributos[atributos_categoricos] = codificador_atributos_discretos.transform(
        atributos[atributos_categoricos]
    )
    return atributos   

def labelEncoderObjetivo(objetivo):
    codificador_objetivo = LabelEncoder()
    objetivo = codificador_objetivo.fit_transform(objetivo)
    return objetivo


def atributosNormalizados(atributos):
    normalizador = MinMaxScaler(
        feature_range=(0,1)
    )
    atributos_normalizados = atributos.copy()
    atributos_normalizados[:] = normalizador.fit_transform(atributos_normalizados)
    return atributos_normalizados


def preprocesamientoDeDatos(nombreArchivocsv, nombreObjetivo):
    atributos, objetivo = separarAtributosObjetivo(nombreArchivocsv, nombreObjetivo)
    atributos = codificacionDeAtributosCategoricos(atributos)
    objetivo = labelEncoderObjetivo(objetivo)
    atributos_normalizados = atributosNormalizados(atributos)
    return atributos, objetivo, atributos_normalizados

def EvaluacionRobusta(algoritmo, datos_entrenamiento, objetivo_entrenamiento, variables, n_exp, k):
    resultados = 0
    datos = datos_entrenamiento[variables]
    for i in range (0, n_exp):
        resultado_validacion = cross_val_score(algoritmo, datos, objetivo_entrenamiento, 
                                      scoring='balanced_accuracy', cv=k, n_jobs=-1)
        resultados += resultado_validacion.mean()  
    return resultados/n_exp


def definirRetorno(vectorMejoresSolucionesTemporales):
    pd.set_option('display.max_colwidth', None)
    retorno= pd.DataFrame(vectorMejoresSolucionesTemporales, 
                          columns=['Solución', 'Rendimiento', 'Tamaño'])

    retorno = retorno.sort_values(by='Rendimiento', ascending=False) 
    return retorno

def ExperimentosConHiperparametrosDeAlgoritmo(busqueda, algoritmo_constructor, atributos_entrenamiento, objetivo_entrenamiento, hiperparametros, n_exp, k, M):
    resultados = []
    
    for params in hiperparametros:
        algoritmo = algoritmo_constructor(**params)
        resultado = busqueda(atributos_entrenamiento, objetivo_entrenamiento, algoritmo, n_exp, k, M)
        resultado['Hiperparámetros'] = str(params)
        resultados.append(resultado)
    
    resultados_totales = pd.concat(resultados, ignore_index=True) 
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    return resultados_totales

def capacidadPredictivaDelConjuntoCompleto(algoritmo_constructor, hiperparametros, atributos_entrenamiento, objetivo_entrenamiento, variablesDataset, n_exp, k):
    for params in hiperparametros:
        algoritmo = algoritmo_constructor(**params)
        resultado = EvaluacionRobusta(algoritmo, atributos_entrenamiento, objetivo_entrenamiento, variablesDataset, n_exp, k)
        
        print(f"Hiperparámetros: {params}")
        print(f"Resultado: {resultado}")
        print("-" * 50)

def obtener_mejor_solucion(dataframe):
    df_ordenado = dataframe.sort_values(by=['Rendimiento', 'Tamaño'], ascending=[False, True])
    mejor_fila = df_ordenado.iloc[0]
    return mejor_fila['Solución'], mejor_fila['Hiperparámetros']