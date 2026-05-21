"""
Caso_casas_GD.py
Regresión Lineal Múltiple — Precio de Vivienda
-----------------------------------------------
Implementación desde cero de Regresión Lineal Multivariable
para predecir el precio de venta de propiedades en USD.
Incluye: normalización Z-Score, Ecuación Normal y
Gradient Descent con visualización de la curva de aprendizaje.
Dataset: 14 propiedades con 4 features (área, habitaciones,
antigüedad y distancia al centro).
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
# FEATURES POLINOMICAS (NO SE)
# ===========================================================================
def expandir_polinomial(datos): # Los datos entran como lista de ([agua, temp], y).
    datos_expandidos = []
    for x_vec in datos:
        area = x_vec[0]
        hab = x_vec[1]
        anios = x_vec[2]
        dist = x_vec[3]
        y = x_vec[4]
        X_matrix = [1, area, hab, anios, dist, y]
        print([1,area,hab,anios,dist])
        datos_expandidos.append(X_matrix)
        
    return datos_expandidos # Retorna  [1, agua, temp]

# ===========================================================================
# NORMALIZACION Z-SCORE
# ===========================================================================
def calcular_estadisticas(datos):
    """
    Calcula media y desviación de cada columna.
    No normaliza columna 0 (bias).
    Sí normaliza y (última columna).
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
    Normaliza TODA la matriz:
    [1, x1, x2, x3, x4, y]
    luego devuelve:
    ([1, x1_norm, ..., x4_norm], y_norm)
    """
    datos_norm = []

    for fila in datos:

        fila_norm = []

        for j in range(len(fila)):
            valor_norm = (fila[j] - medias[j]) / desv[j]
            fila_norm.append(valor_norm)

        x_vec = fila_norm[:-1]
        y     = fila_norm[-1]

        datos_norm.append((x_vec, y))

    return datos_norm

# ===========================================================================
# FUNCIONES BASE
# ===========================================================================
def predecir(x_vec, w):
    """
    f(agua,temp) = w·(agua,temp)  (producto punto)
    f(agua,temp) = w0*1 + w1*agua + w2*temp + w3*agua^2 + w4*temp^2, donde w0 = b
    """
    return sum(w[j] * x_vec[j] for j in range(len(w)))

def calcular_mse_puro(datos, w):
    """MSE puro sin regularización — solo para reportar al final"""
    """MSE = (1/N) * sum((y - y_hat)^2) ->  sin regularización"""
    n = len(datos)
    total = sum((y - predecir(x_vec, w))**2 for x_vec, y in datos)
    return total / n

def calcular_costo(datos, w, lambda_=0.0):
    """MSE + L2: J = (1/N) * sum((y - y_hat)^2) + lambda * sum(wj²) -> con regulaización"""
    regularizacion = lambda_ * sum(wj**2 for wj in w[1:])
    return calcular_mse_puro(datos, w) + regularizacion

def gradiente_mse(datos, w, lambda_):
    """
    dJ/dw_j = (-2/N) * sum(x_ij * (y_i - y_hat_i))
    """
    n      = len(datos)
    n_feat = len(w)
    grad_w = [0.0] * n_feat
    for x_vec, y in datos:
        error = y - predecir(x_vec, w)
        for j in range(n_feat):
            grad_w[j] += -2 * x_vec[j] * error
        
    # Agregar penalización L2 (no al bias w[0])
    grad_w_reg = [gw / n for gw in grad_w]
    for j in range(1, n_feat):
        grad_w_reg[j] += 2 * lambda_ * w[j]
        
    return grad_w_reg

def separador(titulo):
    linea = "=" * 60
    print(f"\n{linea}")
    print(f"  {titulo}")
    print(linea)

# ===========================================================================
# ESTADISTICAS AUXILIARES
# ===========================================================================
def media(lista):
    return sum(lista) / len(lista)

def desviacion_estandar(lista):
    m = media(lista)
    return (sum((xi - m)**2 for xi in lista) / len(lista)) ** 0.5

def calcular_r2(datos_norm, w):
    """R² = 1 - SS_res / SS_tot"""
    ys = [datos_norm[i][1] for i in range(len(datos_norm))]
    y_media = media(ys)
    ss_res  = sum((y - predecir(x, w))**2 for x, y in datos_norm)
    ss_tot  = sum((y - y_media)**2         for y in ys)
    return 1 - ss_res / ss_tot

# ===========================================================================
# ECUACION NORMAL: w = (X⊤X)⁻¹ X⊤y  —  con numpy
# ===========================================================================

def ecuacion_normal(datos_norm):
    """
    Resuelve w = (X⊤X)⁻¹ X⊤y directamente, sin iteraciones.
    Recibe datos_norm: lista de (x_vec, y) ya normalizados.
    Retorna w: lista de coeficientes [w0, w1, w2, w3, w4]
    """

    X = np.array([x_vec for x_vec, _ in datos_norm])
    y = np.array([yi    for _,     yi in datos_norm])

    w = np.linalg.inv(X.T @ X) @ X.T @ y

    return w.tolist()

# ===========================================================================
# GRADIENT DESCENT MULTIVARIABLE
# ===========================================================================
def gradient_descent(datos, lr=0.01, lambda_= 0,
                     iter_max=100000, verbose=True, mostrar_cada=500,
                     pesos_iniciales=None):
    
    n_feat = len(datos[0][0])
    w = pesos_iniciales[:] if pesos_iniciales else [0.0] * n_feat
    
    historial = []   # lista de (iteracion, mse)

    if verbose:
        separador("GRADIENT DESCENT MULTIVARIABLE")
        print(f"  lr={lr}  |  iter_max={iter_max}")
        print(f"  N={len(datos)}  |  features={n_feat}")
        nombres_w = "  ".join(f"{'w'+str(j):>10}" for j in range(n_feat))
        print(f"\n  {'Iter':>7}  {nombres_w}  {'MSE':>12}")
    
        
    for i in range(iter_max):
        
        costo = calcular_costo(datos, w, lambda_)
        historial.append((i, costo))

        if verbose and (i % mostrar_cada == 0 or i < 5):
            vals_w = "  ".join(f"{wj:>10.4f}" for wj in w)
            print(f"  {i:>7}  {vals_w} {costo:>12.6f}")
        
        gw = gradiente_mse(datos, w, lambda_)
        w = [w[j] - lr * gw[j] for j in range(n_feat)]

    costo_final = calcular_costo(datos, w, lambda_)
    historial.append((i + 1, costo_final))

    if verbose:
        vals_w = "  ".join(f"{wj:>10.4f}" for wj in w)
        print(f"  {i+1:>7}  {vals_w}  {costo_final:>12.6f}")
        print(f"\n  Resultado final:")
        # w[0] es el bias, w[1..4] son los pesos
        print(f"    w[0] = {w[0]:.6f}  ← intercepto (bias)")
        print(f"    w[1] = {w[1]:.6f}  ← area")
        print(f"    w[2] = {w[2]:.6f}  ← habitaciones")
        print(f"    w[3] = {w[3]:.6f}  ← anios")
        print(f"    w[4] = {w[4]:.6f}  ← distancia")
        print(f"  MSE final     : {calcular_mse_puro(datos, w):.6f}")  # error real del modelo
        print(f"  Iteraciones realizadas: {i+1}")

    return w, historial

# ===========================================================================
# GRAFICO SENCILLO DE EVOLUCION DEL ERROR (terminal)
# ===========================================================================

def graficar_mse(historial, lr=0.0):
    """
    Grafica la curva de aprendizaje (MSE vs iteraciones)
    usando matplotlib.
    """
    iters = [h[0] for h in historial]
    mses  = [h[1] for h in historial]

    plt.figure(figsize=(10, 5))

    plt.plot(iters, mses, color="red", linewidth=1.5)

    plt.title("Curva de aprendizaje (Gradient Descent)", fontsize=13)
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.grid(True, linestyle="--", alpha=0.4)

    # marcar el punto final
    plt.scatter(iters[-1], mses[-1], color="black", s=40)
    plt.annotate(
        f"Final MSE = {mses[-1]:.6f}",
        xy=(iters[-1], mses[-1]),
        xytext=(-80, 20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="gray")
    )

    if lr:
        plt.title(f"Curva de aprendizaje (lr = {lr})")

    plt.tight_layout()
    plt.savefig("mse_gradient_descent.png", dpi=150)
    plt.show()
    
# ===========================================================================
# MAIN — CONFIGURACION
# ===========================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # CONFIGURACION DE ENTRADA
    # ------------------------------------------------------------------
    # Hiperparametros (modifica estos valores)
    LEARNING_RATE  = 0.01
    ITER_MAX       = 5000               # limite de seguridad
    LAMBDA = 0  # 0 = sin regularización, prueba con 0.001, 0.01, 0.1
    
    # Pesos iniciales: None => todos 0, o lista de 5 floats, ej: [0.5, -0.2, 0.1, 0.3]
    #    Son 5 valores en total por que 1 es para el bias, y 2 para cada parametro (agua, temp)
    # b -> w0, agua -> w1, temp -> w2
    PESOS_INICIALES = [0,0,0,0,0] # valores para w, primer valor corresponde a w0 = bias

    # ------------------------------------------------------------------

    print("\n" + "#"*60)
    print("  REGRESION LINEAL MULTIVARIABLE DESDE CERO")
    print("#"*60)
    
    # Se genera la matriz de X, esta matriz es de 12x3 (Son 12 datos, y 3 parametros: b, agua, temp)
    print(f"\nMatriz X (14x5)")
    print(f"\n")
    datos_casa = expandir_polinomial(datos_casa)
    print(f"\n")
    
    
    # Normalizar
    medias_x, desv_x = calcular_estadisticas(datos_casa)
    
    separador("ESTADISTICAS DE NORMALIZACION (Z-Score)")
    for j in range(len(medias_x)):
        print(f"  Caracteristica {j}: media={medias_x[j]:.4f}  desv={desv_x[j]:.4f}")
    
    datos_norm = normalizar(datos_casa, medias_x, desv_x)
    print(datos_norm)
    
    separador("ECUACION NORMAL")
    w_normal = ecuacion_normal(datos_norm)
    etiquetas = ["intercepto (bias)", "area", "habitaciones", "anios", "distancia"]
    for i, (wi, etiq) in enumerate(zip(w_normal, etiquetas)):
        print(f"  w[{i}] = {wi:.6f}  ← {etiq}")
    print(f"  MSE (ec. normal) : {calcular_mse_puro(datos_norm, w_normal):.6f}")
    print(f"  R²  (ec. normal) : {calcular_r2(datos_norm, w_normal):.6f}")
    
    # Entrenar
    w, historial = gradient_descent(
        datos_norm,
        lr              = LEARNING_RATE,
        lambda_         = LAMBDA,
        iter_max        = ITER_MAX,
        verbose         = True,
        mostrar_cada    = max(1, ITER_MAX // 10),
        pesos_iniciales = PESOS_INICIALES,
    )
    # Calcular r2
    print(f"  r2 para el modelo:  {calcular_r2(datos_norm, w)}")
    
    # Grafico de error
    graficar_mse(historial, lr=LEARNING_RATE)
    
    # ===========================================================================
    # PREDICCION CON NUEVOS DATOS
    # ===========================================================================

    separador("PREDICCION CON NUEVOS DATOS")

    # Ingresa aquí los valores de la nueva casa: [area, habitaciones, anios, distancia]
    nueva_casa = [120, 3, 10, 7]

    # Normalizar con Z-Score usando las mismas estadísticas del entrenamiento
    nueva_casa_norm = normalizar([[1] + nueva_casa + [0]], medias_x, desv_x)
    x_nuevo = nueva_casa_norm[0][0]  # extraer solo el x_vec

    # Predecir en espacio normalizado y desnormalizar
    y_norm_pred  = predecir(x_nuevo, w)
    precio_pred  = y_norm_pred * desv_x[5] + medias_x[5]

    print(f"  Área         : {nueva_casa[0]} m²")
    print(f"  Habitaciones : {nueva_casa[1]}")
    print(f"  Antigüedad   : {nueva_casa[2]} años")
    print(f"  Distancia    : {nueva_casa[3]} km")
    print(f"\n  Precio estimado: ${precio_pred:,.2f}")

    
    
    

