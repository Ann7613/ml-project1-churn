"""
Regresion Lineal Multivariable desde cero: Gradient Descent
Normalizacion: Z-Score | Entrada: CSV | Corte: por error minimo
"""
import math
import matplotlib.pyplot as plt
import numpy as np

datos_casa = [
    [95, 4, 13, 12.5, 171600],
    [123, 2, 14, 8, 216200],
    [118, 2, 22, 15, 196900],
    [96, 4, 8, 5.5, 190200],
    [182, 4, 19, 9, 305700],
    [86, 4, 5, 3.5, 180100],
    [63, 2, 7, 18, 133600],
    [193, 2, 3, 4, 320000],
    [155, 2, 29, 22, 242700],
    [128, 2, 3, 6.5, 216800],
    [195, 5, 18, 7.5, 357400],
    [115, 5, 22, 20, 219800],
    [105, 3, 10, 2, 198500],
    [140, 3, 6, 14, 241300],
]

# ===========================================================================
# NORMALIZACION Z-SCORE
# ===========================================================================
def calcular_estadisticas(datos):
    """
    Calcula media y desviación de cada columna.
    No normaliza columna 0 (bias).
    """
    n = len(datos)
    n_feat = len(datos[0])
    medias = []
    desv = []
    for j in range(n_feat):
        # bias
        if j == 0:
            medias.append(0.0)
            desv.append(1.0)
            continue
        media = sum(datos[i][j] for i in range(n)) / n
        varianza = sum((datos[i][j] - media)**2 for i in range(n)) / n
        sigma = math.sqrt(varianza)
        medias.append(media)
        desv.append(sigma if sigma > 0 else 1.0)
    return medias, desv

def normalizar(datos, medias, desv):
    """
    Normaliza TODA la matriz menos y:
    [1, x1, x2, x3, x4]
    luego devuelve:
    ([1, x1_norm, ..., x4_norm])
    """
    datos_norm = []

    for fila in datos:
        fila_norm = []
        for j in range(len(fila)):
            valor_norm = (fila[j] - medias[j]) / desv[j]
            fila_norm.append(valor_norm)
        datos_norm.append((fila_norm))
    return datos_norm

# ===========================================================================
# REGRESION LOGISTICA — FUNCIONES
# ===========================================================================

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predecir_logistica(x_vec, w):
    z = sum(w[j] * x_vec[j] for j in range(len(w)))
    return sigmoid(z)

def calcular_bce(datos, w):
    """Binary Cross-Entropy: J = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))"""
    n = len(datos)
    total = 0.0
    for x_vec, y in datos:
        p = predecir_logistica(x_vec, w)
        p = max(min(p, 1 - 1e-15), 1e-15)  # evitar log(0)
        total += y * math.log(p) + (1 - y) * math.log(1 - p)
    return -total / n

def calcular_accuracy(datos, w):
    correctos = sum(
        1 for x_vec, y in datos
        if round(predecir_logistica(x_vec, w)) == y
    )
    return correctos / len(datos)

def gradient_descent_logistico(datos, lr=0.01, iter_max=2000,
                                verbose=True, nombre_feature="x"):
    n_feat = len(datos[0][0])
    w = [0.0] * n_feat

    for i in range(iter_max):
        grad = [0.0] * n_feat
        for x_vec, y in datos:
            error = predecir_logistica(x_vec, w) - y
            for j in range(n_feat):
                grad[j] += error * x_vec[j]
        w = [w[j] - lr * grad[j] / len(datos) for j in range(n_feat)]

    if verbose:
        separador(f"MODELO UNIVARIADO — {nombre_feature}")
        print(f"  w[0] = {w[0]:.6f}  ← intercepto (bias)")
        for j in range(1, n_feat):
            print(f"  w[{j}] = {w[j]:.6f}  ← {nombre_feature}")
        print(f"  BCE      : {calcular_bce(datos, w):.6f}")
        print(f"  Accuracy : {calcular_accuracy(datos, w)*100:.1f}%")

    return w

def separador(titulo):
    linea = "=" * 60
    print(f"\n{linea}")
    print(f"  {titulo}")
    print(linea)
    
def gradient_descent_logistico_multi(datos, lr=0.01, iter_max=3000, verbose=True):
    n_feat = len(datos[0][0])
    w = [0.0] * n_feat

    for i in range(iter_max):
        grad = [0.0] * n_feat
        for x_vec, y in datos:
            error = predecir_logistica(x_vec, w) - y
            for j in range(n_feat):
                grad[j] += error * x_vec[j]
        w = [w[j] - lr * grad[j] / len(datos) for j in range(n_feat)]

    if verbose:
        separador("MODELO COMPLETO — 4 features")
        etiquetas = ["intercepto (bias)", "area", "habitaciones", "antigüedad", "distancia"]
        for j in range(n_feat):
            print(f"  w[{j}] = {w[j]:.6f}  ← {etiquetas[j]}")
        print(f"  BCE      : {calcular_bce(datos, w):.6f}")
        print(f"  Accuracy : {calcular_accuracy(datos, w)*100:.1f}%")

    return w

def matriz_confusion(datos, w, nombre="MODELO COMPLETO"):
    TP = TN = FP = FN = 0
    for x_vec, y in datos:
        pred = round(predecir_logistica(x_vec, w))
        if   y == 1 and pred == 1: TP += 1
        elif y == 0 and pred == 0: TN += 1
        elif y == 0 and pred == 1: FP += 1
        elif y == 1 and pred == 0: FN += 1

    separador(f"MATRIZ DE CONFUSION — {nombre}")
    print(f"                    Pred ESTÁNDAR   Pred PREMIUM")
    print(f"  Real ESTÁNDAR  →       TN={TN}            FP={FP}")
    print(f"  Real PREMIUM   →       FN={FN}            TP={TP}")
    print(f"\n  Accuracy  : {(TP+TN)/(TP+TN+FP+FN)*100:.1f}%")
    print(f"  Precision : {TP/(TP+FP) if TP+FP>0 else 0:.4f}")
    print(f"  Recall    : {TP/(TP+FN) if TP+FN>0 else 0:.4f}")

    # Grafico bonito con matplotlib
    matriz = [[TN, FP], [FN, TP]]
    etiquetas = ["ESTÁNDAR", "PREMIUM"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matriz, cmap="Blues")

    ax.set_xticks([0, 1]); ax.set_xticklabels(etiquetas)
    ax.set_yticks([0, 1]); ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Predicho", fontsize=11)
    ax.set_ylabel("Real",     fontsize=11)
    ax.set_title(f"Matriz de Confusión — {nombre}", fontsize=12)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matriz[i][j]),
                    ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if matriz[i][j] > max(TN,TP,FP,FN)/2 else "black")

    plt.tight_layout()
    plt.savefig("matriz_confusion.png", dpi=150)
    plt.show()

    return matriz
# ===========================================================================
# MAIN — CONFIGURACION
# ===========================================================================

if __name__ == "__main__":

    print("\n" + "#"*60)
    print("  REGRESION LOGISTICA — CLASIFICACION BINARIA")
    print("#"*60)

    # Crear variable binaria
    y_bin = [1 if fila[4] > 200000 else 0 for fila in datos_casa]

    n_premium  = sum(y_bin)
    n_estandar = len(y_bin) - n_premium

    separador("VARIABLE OBJETIVO BINARIA")
    print(f"  PREMIUM  (y=1, precio > $200,000): {n_premium}  propiedades")
    print(f"  ESTÁNDAR (y=0, precio ≤ $200,000): {n_estandar} propiedades")
    print(f"  Total                            : {len(y_bin)}")
    print(f"\n  Ratio: {n_premium}/{n_estandar}", end="  →  ")

    if 0.4 <= n_premium / len(y_bin) <= 0.6:
        print("Dataset balanceado ✓")
    else:
        print("Dataset DESBALANCEADO ⚠")
    
    # Solo las X, sin y_bin
    datos_solo_x = [
        [1, datos_casa[i][0], datos_casa[i][1], datos_casa[i][2], datos_casa[i][3], 0]
        for i in range(len(datos_casa))
    ]
    # Calcular estadisticas
    medias_log, desv_log = calcular_estadisticas(datos_solo_x)
    datos_norm_x = normalizar(datos_solo_x, medias_log, desv_log)

    # Reincorporar y_bin (reemplaza el y normalizado por el binario real y_bin)
    datos_norm = [(fila[:-1], y_bin[i]) for i, fila in enumerate(datos_norm_x)]

    
    # ==============================================
    # Modelos univariados: [bias, feature]
    # ===============================================
    datos_area = [([x_vec[0], x_vec[1]], y) for x_vec, y in datos_norm]
    datos_hab  = [([x_vec[0], x_vec[2]], y) for x_vec, y in datos_norm]
    datos_anio = [([x_vec[0], x_vec[3]], y) for x_vec, y in datos_norm]
    datos_dist = [([x_vec[0], x_vec[4]], y) for x_vec, y in datos_norm]

    w_area = gradient_descent_logistico(datos_area, lr=0.01, iter_max=2000, verbose=False, nombre_feature="area")
    w_area
    w_hab  = gradient_descent_logistico(datos_hab,  lr=0.01, iter_max=2000, verbose=False, nombre_feature="habitaciones")
    w_hab
    w_anio = gradient_descent_logistico(datos_anio, lr=0.01, iter_max=2000, verbose=False, nombre_feature="antigüedad")
    w_anio
    w_dist = gradient_descent_logistico(datos_dist, lr=0.01, iter_max=2000, verbose=False, nombre_feature="distancia")
    w_dist
    
    # ==============================================
    # Modelo multivariados: [bias, x1 x2, x3, x4]
    # ===============================================
    
    w_multi = gradient_descent_logistico_multi(datos_norm, lr=0.01, iter_max=3000)
    
    separador("Comparación de accurancies")
    modelos = [
        ("Solo area",         datos_area, w_area),
        ("Solo habitaciones", datos_hab,  w_hab),
        ("Solo antiguedad",   datos_anio, w_anio),
        ("Solo distancia",    datos_dist, w_dist),
        ("Modelo completo",   datos_norm, w_multi),
    ]
    for nombre, datos_m, w_m in modelos:
        acc = calcular_accuracy(datos_m, w_m) * 100
        print(f"  {nombre:<22}: {acc:.1f}%")
    
    # ==============================================
    # MATRIZ DE CONFUSION
    # ===============================================
    
    matriz_confusion(datos_norm, w_multi, nombre="Modelo logístico completo")
    
    
    # ==============================================
    # Predicción nueva propiedad
    # ===============================================
    
    separador("PREDICCION NUEVA PROPIEDAD")
    nueva = [175, 4, 8, 6]

    nueva_norm = normalizar(
        [[1, nueva[0], nueva[1], nueva[2], nueva[3], 0]],
        medias_log, desv_log
    )
    x_nueva = nueva_norm[0][:-1]
    p_premium = predecir_logistica(x_nueva, w_multi)
    clase = "PREMIUM" if p_premium >= 0.5 else "ESTÁNDAR"

    print(f"  Área         : {nueva[0]} m²")
    print(f"  Habitaciones : {nueva[1]}")
    print(f"  Antigüedad   : {nueva[2]} años")
    print(f"  Distancia    : {nueva[3]} km")
    print(f"\n  P(PREMIUM)     = {p_premium:.4f}")
    print(f"  P(ESTÁNDAR)    = {1-p_premium:.4f}")
    print(f"  Clase predicha : {clase}")