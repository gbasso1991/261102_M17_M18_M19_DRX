#%% DRX en polvo de muestras M1z
# Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
import matplotlib as mpl
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import minimum_filter, gaussian_filter
#%%%
# #%% Williamson-Hall ponderado por incertidumbre del FWHM
# def williamson_hall(tabla, nombre, columna_fwhm='FWHM con (°)',
#                     columna_err='Error FWHM con (°)'):
#     columnas = ['Reflexión', '2theta (°)', columna_fwhm, columna_err]
#     datos = tabla[columnas].replace([np.inf, -np.inf], np.nan).dropna()
#     datos = datos[
#         (datos[columna_fwhm] > 0) &
#         (datos[columna_err] > 0)
#     ].copy()

#     if len(datos) < 3:
#         raise ValueError(f'{nombre}: se necesitan al menos 3 reflexiones válidas con incertidumbres')

#     theta = np.deg2rad(datos['2theta (°)'].to_numpy()/2)
#     beta = np.deg2rad(datos[columna_fwhm].to_numpy())
#     sigma_beta = np.deg2rad(datos[columna_err].to_numpy())

#     x = 4*np.sin(theta)
#     y = beta*np.cos(theta)
#     sigma_y = sigma_beta*np.cos(theta)

#     validos = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma_y) & (sigma_y > 0)
#     x, y, sigma_y = x[validos], y[validos], sigma_y[validos]
#     datos = datos.iloc[np.flatnonzero(validos)]

#     if len(datos) < 3:
#         raise ValueError(f'{nombre}: quedan menos de 3 reflexiones válidas')

#     def recta(x, pendiente, ordenada):
#         return pendiente*x + ordenada

#     parametros, cov = curve_fit(
#         recta, x, y,
#         sigma=sigma_y,
#         absolute_sigma=True
#     )
#     pendiente, ordenada = parametros
#     err_pendiente, err_ordenada = np.sqrt(np.maximum(np.diag(cov), 0))

#     y_fit = recta(x, pendiente, ordenada)
#     chi2_red = np.sum(((y-y_fit)/sigma_y)**2)/(len(y)-2)
#     R2 = 1-np.sum((y-y_fit)**2)/np.sum((y-np.mean(y))**2)

#     lambda_nm = lambda_X/10
#     if ordenada > 0:
#         D = K*lambda_nm/ordenada
#         err_D = abs(K*lambda_nm/ordenada**2)*err_ordenada
#     else:
#         D, err_D = np.nan, np.nan
#         print(f'Advertencia: {nombre}, ordenada al origen no positiva.')

#     epsilon = pendiente
#     err_epsilon = err_pendiente

#     print(f'{nombre} - Williamson-Hall ponderado ({columna_fwhm})')
#     print(f'D = {D:.2f} ± {err_D:.2f} nm')
#     print(f'Microdeformación ε = {epsilon:.3e} ± {err_epsilon:.3e}')
#     print(f'Ordenada al origen = {ordenada:.5e} ± {err_ordenada:.2e}')
#     print(f'χ² reducido = {chi2_red:.3f}')
#     print(f'R² = {R2:.4f}')
#     print(f'N reflexiones = {len(datos)}')

#     fig, ax = plt.subplots(figsize=(7,5), constrained_layout=True)
#     ax.errorbar(x, y, yerr=sigma_y, fmt='o', capsize=3, ms=6,
#                 label='Reflexiones')
#     xfit = np.linspace(x.min(), x.max(), 200)
#     ax.plot(xfit, recta(xfit, *parametros), '-',
#             label=f'Ajuste ponderado: $R^2$ = {R2:.3f}')
#     for xi, yi, (_, fila) in zip(x, y, datos.iterrows()):
#         ax.annotate(fila['Reflexión'], (xi, yi),
#                     xytext=(4,4), textcoords='offset points', fontsize=8)

#     ax.set_xlabel(r'$4\sin\theta$')
#     ax.set_ylabel(r'$\beta\cos\theta$ (rad)')
#     ax.set_title(f'{nombre}: Williamson-Hall ponderado')
#     ax.grid(alpha=0.3)
#     ax.legend()
#     plt.show()

#     return {
#         'Muestra': nombre,
#         'FWHM utilizado': columna_fwhm,
#         'Incertidumbre FWHM': columna_err,
#         'D (nm)': D,
#         'Error D (nm)': err_D,
#         'Microdeformación': epsilon,
#         'Error microdeformación': err_epsilon,
#         'Ordenada': ordenada,
#         'Error ordenada': err_ordenada,
#         'chi2 reducido': chi2_red,
#         'R²': R2,
#         'N reflexiones': len(datos)
#     }
#%%pvoigt_fondo
def pvoigt_fondo(x, A, xc, fwhm, eta, b0, b1):
    """
    Calcula un perfil pseudo-Voigt sobre un fondo lineal.

    Parámetros
    ----------
    x : array_like
        Posiciones angulares 2theta, en grados.
    A : float
        Amplitud del pico respecto del fondo.
    xc : float
        Posición del centro del pico, en grados de 2theta.
    fwhm : float
        Anchura a media altura del pico (FWHM), en grados.
    eta : float
        Fracción lorentziana del perfil, entre 0 y 1.
        eta = 0 corresponde a una gaussiana y eta = 1
        a una lorentziana.
    b0 : float
        Intensidad del fondo en xc.
    b1 : float
        Pendiente del fondo lineal, en unidades de intensidad
        por grado.

    Retorna
    -------
    array_like
        Intensidad calculada para cada posición angular x.
    """
    z = (x-xc)/fwhm
    gauss = np.exp(-4*np.log(2)*z**2)
    lorentz = 1/(1+4*z**2)
    perfil = (1-eta)*gauss + eta*lorentz
    return b0 + b1*(x-xc) + A*perfil

#%% analizar_pico
def analizar_pico(x, y, centro_ref, ventana=0.9, picos_detectados=None,
                  correcion_LaB6=False, df_LaB6=None,
                  K=0.9, lambda_X=1.5406, D_LaB6=800):
    """
    Ajusta un pico mediante una pseudo-Voigt sobre fondo lineal y,
    opcionalmente, estima el FWHM corregido por ensanchamiento instrumental.

    Parámetros adicionales
    ----------------------
    correcion_LaB6 : bool
        Si True, estima y aplica la corrección instrumental a partir de LaB6.
    df_LaB6 : DataFrame
        Tabla de calibración LaB6, con columnas:
        '2theta Ka1 (°)', 'FWHM Ka1 (°)' y 'eta'.
    K : float
        Factor de Scherrer.
    lambda_X : float
        Longitud de onda en Å.
    D_LaB6 : float
        Tamaño de dominio de referencia de LaB6, en nm.

    Retorna
    -------
    xp : ndarray
        Posiciones angulares usadas en el ajuste.
    popt : ndarray
        Parámetros ajustados [A, xc, fwhm, eta, b0, b1].
    pcov : ndarray
        Matriz de covarianza del ajuste.
    fwhm_corregido : float
        FWHM corregido en grados de 2theta. Si no se activa la
        corrección, coincide con el FWHM ajustado.
    """
    mask = (x >= centro_ref-ventana) & (x <= centro_ref+ventana)
    xp, yp = x[mask], y[mask]
    if len(xp) < 10:
        raise ValueError('Pocos puntos para ajustar el pico')

    b0 = np.percentile(yp, 10)
    fwhm0 = max(3*np.median(np.diff(xp)), 0.15)

    candidatos = []
    if picos_detectados is not None:
        candidatos = [
            (indice, theta, intensidad)
            for indice, theta, intensidad in picos_detectados
            if centro_ref-ventana/2 <= theta <= centro_ref+ventana/2
        ]

    if candidatos:
        indice, xc0, intensidad_max = min(
            candidatos, key=lambda pico: abs(pico[1]-centro_ref)
        )
        A0 = max(intensidad_max-b0, 1e-6)
    else:
        imax = np.argmax(yp)
        xc0 = xp[imax]
        A0 = max(yp[imax]-b0, 1e-6)

    popt, pcov = curve_fit(
        pvoigt_fondo, xp, yp,
        p0=[A0, xc0, fwhm0, 0.5, b0, 0],
        bounds=(
            [0, centro_ref-ventana/2, 0.005, 0, -np.inf, -np.inf],
            [np.inf, centro_ref+ventana/2, 2*ventana, 1, np.inf, np.inf]
        ),
        maxfev=30000
    )

    fwhm = popt[2]
    fwhm_corregido = fwhm

    if correcion_LaB6:
        if df_LaB6 is None:
            raise ValueError('Debes pasar df_LaB6 para corregir el ancho')

        # Anchos e intensidades de mezcla de la muestra
        eta_muestra = np.clip(popt[3], 0, 1)
        theta = np.deg2rad(popt[1]/2)

        # Interpolación de la calibración en la posición del pico
        df_cal = df_LaB6.sort_values('2theta Ka1 (°)')
        x_cal = df_cal['2theta Ka1 (°)'].to_numpy()
        beta_cal = df_cal['FWHM Ka1 (°)'].to_numpy()
        eta_cal = df_cal['eta'].to_numpy()

        if not x_cal.min() <= popt[1] <= x_cal.max():
            raise ValueError(
                'El pico está fuera del intervalo angular calibrado por LaB6'
            )

        beta_inst_obs = np.interp(popt[1], x_cal, beta_cal)
        eta_inst = np.clip(np.interp(popt[1], x_cal, eta_cal), 0, 1)

        # Ensanchamiento intrínseco aproximado de LaB6:
        # beta_size = K*lambda/(D*cos(theta)), en radianes de 2theta.
        beta_LaB6_intr = np.rad2deg(
            K*(lambda_X/(D_LaB6*10))*1/np.cos(theta)
        )

        # Descomposición heurística de los FWHM pseudo-Voigt.
        # No es una descomposición exacta de la función instrumental.
        beta_G_inst = beta_inst_obs*np.sqrt(1-eta_inst)
        beta_L_inst_obs = beta_inst_obs*eta_inst
        beta_L_inst = max(beta_L_inst_obs-beta_LaB6_intr, 0)

        beta_G_muestra = fwhm*np.sqrt(1-eta_muestra)
        beta_L_muestra = fwhm*eta_muestra

        # Deconvolución aproximada: gaussiana en cuadratura,
        # lorentziana por resta lineal.
        beta_G_corr = np.sqrt(max(beta_G_muestra**2-beta_G_inst**2, 0))
        beta_L_corr = max(beta_L_muestra-beta_L_inst, 0)

        # FWHM equivalente de Voigt (aproximación de Thompson-Cox-Hastings)
        fwhm_corregido = (
            0.5346*beta_L_corr
            + np.sqrt(0.2166*beta_L_corr**2 + beta_G_corr**2)
        )

    return xp, popt, pcov, fwhm_corregido
#%% promedio_cristalita
# def promedio_cristalita(tabla, columna_D, columna_err):
#     datos = tabla[[columna_D, columna_err]].dropna()
#     datos = datos[
#         np.isfinite(datos[columna_D]) &
#         np.isfinite(datos[columna_err]) &
#         (datos[columna_err] > 0) &
#         (datos[columna_D] > 0)
#     ]

#     D = datos[columna_D].to_numpy()
#     err = datos[columna_err].to_numpy()

#     if len(D) < 2:
#         raise ValueError('Se necesitan al menos dos reflexiones válidas')

#     pesos = 1/err**2
#     D_media = np.sum(pesos*D)/np.sum(pesos)
#     err_media = np.sqrt(1/np.sum(pesos))

#     chi2_red = np.sum(((D-D_media)/err)**2)/(len(D)-1)

#     # Inflar la incertidumbre si la dispersión supera la esperada
#     err_final = err_media*np.sqrt(max(chi2_red, 1))

#     return {
#         'D medio (nm)': D_media,
#         'Incertidumbre (nm)': err_final,
#         'Desviación estándar (nm)': np.std(D, ddof=1),
#         'chi2 reducido': chi2_red,
#         'N reflexiones': len(D)}

# #%% Función general de análisis DRX por muestra
# def analizar_muestra(nombre, dostheta, intensity, picos_detectados):
#     picos = picos_detectados
#     # A partir de acá, mantené el tratamiento de ajuste,
#     # cálculo de Scherrer, tabla y gráficos de la función anterior.
    
#     # Ajuste individual de las reflexiones
#     resultados = []
#     fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
#     ax.plot(dostheta, intensity, '-', color='0.4', lw=0.9, label=nombre)

#     for hkl, centro_ref in reflexiones.items():
#         if not any(abs(theta-centro_ref) <= 0.9/2
#                for _, theta, intens in picos_detectados):
#             continue
#         try:
#             # Ajuste sin corrección instrumental
#             xp, popt, pcov, fwhm_sin = analizar_pico(
#                 dostheta, intensity, centro_ref,
#                 picos_detectados=picos,
#                 correcion_LaB6=False
#             )
#             A, xc, fwhm_deg, eta, b0, b1 = popt

#             # Corrección instrumental LaB6
#             _, popt_corr, pcov_corr, fwhm_con = analizar_pico(
#                 dostheta, intensity, centro_ref,
#                 picos_detectados=picos,
#                 correcion_LaB6=True,
#                 df_LaB6=df_LaB6
#             )

#             theta_rad = np.deg2rad(xc/2)
#             beta_sin = np.deg2rad(fwhm_sin)
#             beta_con = np.deg2rad(fwhm_con)

#             D_sin = K*lambda_X/(beta_sin*np.cos(theta_rad))/10
#             D_con = K*lambda_X/(beta_con*np.cos(theta_rad))/10 if beta_con > 0 else np.nan

#             # Incertidumbres del FWHM y propagación a Scherrer
#             fwhm_err_sin = np.sqrt(max(pcov[2,2], 0))

#             fwhm_err_con = incertidumbre_fwhm_corregido_mc(
#                 popt_corr, pcov_corr, df_LaB6,
#                 K=K, lambda_X=lambda_X, D_LaB6=800, n_mc=5000
#             )

#             D_err_sin = D_sin*fwhm_err_sin/fwhm_sin if fwhm_sin > 0 else np.nan
#             D_err_con = D_con*fwhm_err_con/fwhm_con if fwhm_con > 0 and np.isfinite(fwhm_err_con) else np.nan
            
#             resultados.append({
#                 'Reflexión': hkl,
#                 '2theta teo (°)': centro_ref,
#                 '2theta (°)': xc,
#                 'FWHM sin (°)': fwhm_sin,
#                 'Error FWHM sin (°)': fwhm_err_sin,
#                 'FWHM con (°)': fwhm_con,
#                 'Error FWHM con (°)': fwhm_err_con,
#                 'eta': eta,
#                 'D sin (nm)': D_sin,
#                 'Error sin (nm)': D_err_sin,
#                 'D con (nm)': D_con,
#                 'Error con (nm)': D_err_con
#                         })

#             xfit = np.linspace(xp.min(), xp.max(), 300)
#             ax.plot(xfit, pvoigt_fondo(xfit, *popt), lw=1.2,
#                     label=f'{hkl}: {xc:.2f}°')

#         except (ValueError, RuntimeError) as e:
#             print(f'{hkl}: no se pudo ajustar ({e})')

#     ax.set_xlim(20,80)
#     ax.set_title('M1 - Difractograma de RX')
#     ax.set_ylim(0,None)
#     ax.set_xlabel(r'$2\theta$ (°)')
#     ax.set_ylabel('Intensidad (u.a.)')
#     ax.set_title(f'{nombre}: ajuste pseudo-Voigt en reflexiones de magnetita')
#     ax.grid()
#     ax.legend(ncol=1, fontsize=8)
#     plt.show()

#     tabla = pd.DataFrame(resultados)
#     if tabla.empty:
#         raise ValueError(f'{nombre}: no se pudo ajustar ninguna reflexión')

#     display(tabla.round(5))

#     # Resumen estadístico
#     resumen_sin = promedio_cristalita(tabla, 'D sin (nm)', 'Error sin (nm)')
#     resumen_con = promedio_cristalita(tabla, 'D con (nm)', 'Error con (nm)')

#     print(
#         f"{nombre} sin corrección: D = {resumen_sin['D medio (nm)']:.1f} "
#         f"± {resumen_sin['Incertidumbre (nm)']:.1f} nm; "
#         f"desviación estándar = {resumen_sin['Desviación estándar (nm)']:.1f} nm; "
#         f"χ² reducido = {resumen_sin['chi2 reducido']:.1f}; "
#         f"N = {resumen_sin['N reflexiones']}"
#     )
#     print(
#         f"{nombre} con corrección: D = {resumen_con['D medio (nm)']:.1f} "
#         f"± {resumen_con['Incertidumbre (nm)']:.1f} nm; "
#         f"desviación estándar = {resumen_con['Desviación estándar (nm)']:.1f} nm; "
#         f"χ² reducido = {resumen_con['chi2 reducido']:.1f}; "
#         f"N = {resumen_con['N reflexiones']}"
#     )

#     # Comparación de D frente a 2theta
#     fig2, ax = plt.subplots(figsize=(9,5), constrained_layout=True)

#     x = tabla['2theta (°)'].to_numpy()
#     D_sin = tabla['D sin (nm)'].to_numpy()
#     err_sin = tabla['Error sin (nm)'].to_numpy()
#     D_con = tabla['D con (nm)'].to_numpy()
#     err_con = tabla['Error con (nm)'].to_numpy()

#     orden = np.argsort(x)
#     x, D_sin, err_sin, D_con, err_con = (
#         v[orden] for v in (x, D_sin, err_sin, D_con, err_con)
#     )

#     ax.errorbar(x, D_sin, yerr=err_sin, fmt='o-', capsize=3,
#                 lw=1.2, ms=5, label='Sin corrección instrumental')
#     ax.errorbar(x, D_con, yerr=err_con, fmt='s--', capsize=3,
#                 lw=1.2, ms=5, label='Con corrección LaB$_6$')

#     ax.axhspan(
#         resumen_sin['D medio (nm)'] - resumen_sin['Desviación estándar (nm)'],
#         resumen_sin['D medio (nm)'] + resumen_sin['Desviación estándar (nm)'],
#         color='C0', alpha=0.10
#     )
#     ax.axhline(
#         resumen_sin['D medio (nm)'], color='C0', ls=':', lw=1.5,
#         label=f"Media sin corrección: {resumen_sin['D medio (nm)']:.1f} ± {resumen_sin['Desviación estándar (nm)']:.1f} nm"
#     )

#     ax.axhspan(
#         resumen_con['D medio (nm)'] - resumen_con['Desviación estándar (nm)'],
#         resumen_con['D medio (nm)'] + resumen_con['Desviación estándar (nm)'],
#         color='C1', alpha=0.10
#     )
#     ax.axhline(
#         resumen_con['D medio (nm)'], color='C1', ls=':', lw=1.5,
#         label=f"Media con corrección: {resumen_con['D medio (nm)']:.1f} ± {resumen_con['Desviación estándar (nm)']:.1f} nm"
#     )

#     for _, fila in tabla.iterrows():
#         ax.annotate(
#             fila['Reflexión'],
#             (fila['2theta (°)'], fila['D sin (nm)']),
#             xytext=(0,-15), textcoords='offset points',
#             ha='center', fontsize=8
#         )

#     ax.set_xlabel(r'$2\theta$ (°)')
#     ax.set_ylabel('Tamaño de cristalita, $D$ (nm)')
#     ax.set_title(f'{nombre}: comparación de tamaños de cristalita por Scherrer')
#     ax.set_xlim(28,82)
#     ax.set_ylim(0,)
#     ax.grid(alpha=0.3)
#     ax.legend(fontsize=8)
#     plt.show()

#     return tabla, resumen_sin, resumen_con, picos, fig,fig2
# #%% Parámetros para Análisis de Scherrer: pseudo-Voigt + fondo lineal
# K = 0.9
# lambda_X = 1.5406  # Å, Cu Kα
# beta_inst_deg = None  # Sin patrón de referencia: no se corrige

# # Posiciones aproximadas de las reflexiones de magnetita (2theta)
# reflexiones = {
#     '(220)': 30.1,
#     '(311)': 35.5,
#     '(400)': 43.1,
#     '(422)': 53.4,
#     '(511)': 57.0,
#     '(440)': 62.6,
#     '(620)': 71.0,
#     '(533)': 74.0,
#     '(444)': 79.0}

# #%% Patrón de calibración LaB6
# data_LaB6 = np.loadtxt('data_DRX/LAB6PHI2.ASC', skiprows=0)
# dostheta_LaB6 = data_LaB6[:, 0]
# intensity_LaB6 = data_LaB6[:, 1]

# # Visualización del difractograma
# fig_LaB6a, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
# ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.3', lw=0.9, label='LaB$_6$')
# ax.set_xlim(20,85)
# ax.set_ylim(0, None)
# ax.set_xlabel(r'$2\theta$ (°)')
# ax.set_ylabel('Intensidad (u.a.)')
# ax.set_title('Patrón de calibración LaB$_6$')
# ax.grid()
# ax.legend()
# ax.set_xlabel('2θ (°)')
# plt.show()


# #% Detección de picos del patrón LaB6
# prominencia_LaB6 = 0.01*np.ptp(intensity_LaB6)
# peaks_LaB6, props_LaB6 = find_peaks(
#     intensity_LaB6,prominence=prominencia_LaB6,
#     width=2,    distance=5)

# picos_LaB6 = list(zip(peaks_LaB6,dostheta_LaB6[peaks_LaB6],intensity_LaB6[peaks_LaB6]))

# fig_LaB6b, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
# ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.3', lw=0.9, label='LaB$_6$')
# ax.plot(dostheta_LaB6[peaks_LaB6], intensity_LaB6[peaks_LaB6], 'r.', ms=6, label='Picos detectados')
# ax.set_xlim(20, 85)
# ax.set_ylim(0, None)
# ax.set_xlabel(r'$2\theta$ (°)')
# ax.set_ylabel('Intensidad (u.a.)')
# ax.set_title('Detección de máximos del patrón LaB$_6$')
# ax.grid()
# ax.legend()
# ax.set_xlabel('2θ (°)')
# plt.show()

# print(f'LaB6: {len(peaks_LaB6)} picos detectados\n')
# print('indx |  2θ (°) | Intensidad')
# for i, theta, intens in picos_LaB6:
#     print(f'{i:4} | {theta:7.2f} | {intens:10.0f}')


# #%% Ajuste de dobletes LaB6: pseudo-Voigt + fondo lineal
# lambda_Ka1 = 1.5406
# lambda_Ka2 = 1.5444
# rel_int_Ka2 = 0.5
# a_ref_LaB6 = 4.156826  # Å, NIST SRM 660c; usar solo si corresponde al patrón

# # Reflexiones: índice, N=h²+k²+l² y máximo principal detectado (Kalpha1)
# reflexiones_LaB6 = [
#     ('(100)', 1, 21.34),
#     ('(110)', 2, 30.38),
#     ('(111)', 3, 37.42),
#     ('(200)', 4, 43.50),
#     ('(210)', 5, 48.94),
#     ('(211)', 6, 53.96),
#     ('(220)', 8, 63.20),
#     ('(300)/(221)', 9, 67.54),
#     ('(310)', 10, 71.72),
#     ('(311)', 11, 75.82),
#     ('(222)', 12, 79.84),
#     ('(320)', 13, 83.82)
# ]

# # def doblete_pvoigt_fondo(x, A, xc, fwhm, eta, b0, b1):
# #     # xc es la posición 2theta de Kalpha1
# #     theta1 = np.deg2rad(xc/2)
# #     argumento = (lambda_Ka2/lambda_Ka1)*np.sin(theta1)
# #     theta2 = np.arcsin(np.clip(argumento, -1, 1))
# #     xc2 = np.rad2deg(2*theta2)
# #     z1 = (x-xc)/fwhm
# #     z2 = (x-xc2)/fwhm
# #     g1 = np.exp(-4*np.log(2)*z1**2)
# #     l1 = 1/(1+4*z1**2)
# #     g2 = np.exp(-4*np.log(2)*z2**2)
# #     l2 = 1/(1+4*z2**2)
# #     perfil1 = (1-eta)*g1 + eta*l1
# #     perfil2 = (1-eta)*g2 + eta*l2
# #     return b0 + b1*(x-xc) + A*(perfil1 + rel_int_Ka2*perfil2)

# def doblete_pvoigt_fondo(x, A, xc, fwhm, eta, rel_int_Ka2, b0, b1):
#     theta1 = np.deg2rad(xc/2)
#     argumento = (lambda_Ka2/lambda_Ka1)*np.sin(theta1)
#     theta2 = np.arcsin(np.clip(argumento, -1, 1))
#     xc2 = np.rad2deg(2*theta2)

#     z1 = (x-xc)/fwhm
#     z2 = (x-xc2)/fwhm
#     g1 = np.exp(-4*np.log(2)*z1**2)
#     l1 = 1/(1+4*z1**2)
#     g2 = np.exp(-4*np.log(2)*z2**2)
#     l2 = 1/(1+4*z2**2)

#     perfil1 = (1-eta)*g1 + eta*l1
#     perfil2 = (1-eta)*g2 + eta*l2

#     return b0 + b1*(x-xc) + A*(perfil1 + rel_int_Ka2*perfil2)

# resultados_LaB6 = []
# ajustes_LaB6 = {}

# for hkl, N, centro_ref in reflexiones_LaB6:
#     # Ventana local alrededor de la reflexión
#     ventana = 0.55
#     mask = (dostheta_LaB6 >= centro_ref-ventana) & (dostheta_LaB6 <= centro_ref+ventana)
#     xp = dostheta_LaB6[mask]
#     yp = intensity_LaB6[mask]
#     if len(xp) < 10:
#         print(f'{hkl}: no hay suficientes puntos para ajustar')
#         continue

#     # Inicialización usando el máximo detectado más cercano
#     candidatos = [
#         (i, theta, intens) for i, theta, intens in picos_LaB6
#         if centro_ref-0.25 <= theta <= centro_ref+0.25
#     ]
#     if candidatos:
#         _, xc0, intens0 = min(candidatos, key=lambda p: abs(p[1]-centro_ref))
#     else:
#         xc0 = xp[np.argmax(yp)]
#         intens0 = np.max(yp)

#     b0 = np.percentile(yp, 10)
#     A0 = max(intens0-b0, 1e-6)
#     fwhm0 = 0.07

#     try:
#         popt, pcov = curve_fit(
#             doblete_pvoigt_fondo, xp, yp,
#             p0=[A0, xc0, fwhm0, 0.5, 0.5, b0, 0],
#             bounds=(
#                 [0, centro_ref-0.25, 0.005, 0, 0, -np.inf, -np.inf],
#                 [np.inf, centro_ref+0.25, 0.30, 1, 1, np.inf, np.inf]
#             ),maxfev=50000)

#         A, xc, fwhm, eta, rel_int_Ka2, b0_fit, b1 = popt        
#         errores = np.sqrt(np.maximum(np.diag(pcov), 0))
#         err_xc = errores[1]
#         err_fwhm = errores[2]

#         theta1 = np.deg2rad(xc/2)
#         theta2 = np.arcsin(np.clip(lambda_Ka2/lambda_Ka1*np.sin(theta1), -1, 1))
#         xc2 = np.rad2deg(2*theta2)

#         # Parámetro de red calculado con Kalpha1
#         a = lambda_Ka1*np.sqrt(N)/(2*np.sin(theta1))
#         err_a = abs(a/np.tan(theta1))*np.deg2rad(err_xc/2)

#         resultados_LaB6.append({
#             'Reflexión': hkl,
#             'N': N,
#             '2theta Ka1 (°)': xc,
#             '2theta Ka2 (°)': xc2,
#             'FWHM Ka1 (°)': fwhm,
#             'Error FWHM (°)': err_fwhm,
#             'eta': eta,
#             'Error eta': errores[3],
#             'a (Å)': a,
#             'Error centro (°)': err_xc,
#             'Error a (Å)': err_a})
        
#         ajustes_LaB6[hkl] = (xp, yp, popt)

#     except (RuntimeError, ValueError) as e:
#         print(f'{hkl}: fallo el ajuste ({e})')

# df_LaB6 = pd.DataFrame(resultados_LaB6)
# display(df_LaB6.round(6))
# #%%
# #%% Incertidumbre del FWHM corregido por Monte Carlo
# def incertidumbre_fwhm_corregido_mc(popt, pcov, df_LaB6,
#                                     K=0.9, lambda_X=1.5406,
#                                     D_LaB6=800, n_mc=10000, seed=None):
#     """
#     Estima por Monte Carlo la incertidumbre del FWHM corregido por
#     ensanchamiento instrumental mediante la calibración con LaB6.

#     Genera realizaciones aleatorias de los parámetros del ajuste de la
#     muestra a partir de su matriz de covarianza. Para cada realización,
#     interpola el FWHM y la fracción lorentziana de la calibración
#     instrumental, muestrea sus incertidumbres y aplica la misma
#     corrección aproximada utilizada en analizar_pico().

#     La incertidumbre del FWHM corregido se estima como la desviación
#     estándar muestral de los anchos corregidos obtenidos en las
#     realizaciones válidas.

#     Parámetros
#     ----------
#     popt : array_like
#         Parámetros ajustados del perfil pseudo-Voigt de la muestra,
#         en el orden [A, xc, fwhm, eta, b0, b1].
#     pcov : ndarray
#         Matriz de covarianza de los parámetros ajustados.
#     df_LaB6 : pandas.DataFrame
#         Tabla de calibración instrumental. Debe contener las columnas
#         '2theta Ka1 (°)', 'FWHM Ka1 (°)', 'Error FWHM (°)', 'eta' y,
#         opcionalmente, 'Error eta'.
#     K : float
#         Factor de forma de Scherrer.
#     lambda_X : float
#         Longitud de onda de rayos X, en Å.
#     D_LaB6 : float
#         Tamaño de dominio de referencia de LaB6, en nm.
#     n_mc : int
#         Número de realizaciones Monte Carlo.
#     seed : int or None
#         Semilla del generador aleatorio para reproducibilidad.

#     Retorna
#     -------
#     float
#         Desviación estándar de los FWHM corregidos, en grados de 2theta.
#         Devuelve np.nan si hay insuficientes realizaciones válidas.

#     Limitaciones
#     ------------
#     La propagación supone parámetros aproximadamente gaussianos y trata
#     como independientes las incertidumbres interpoladas del FWHM y eta
#     instrumentales. No incorpora las covarianzas de los parámetros de
#     calibración ni las incertidumbres sistemáticas de la corrección.
#     """
#     rng = np.random.default_rng(seed)

#     cov = np.asarray(pcov, dtype=float)
#     if not np.all(np.isfinite(cov)):
#         return np.nan

#     cov = (cov+cov.T)/2
#     valores, vectores = np.linalg.eigh(cov)
#     raiz_cov = vectores @ np.diag(np.sqrt(np.maximum(valores, 0)))

#     draws = np.asarray(popt) + rng.standard_normal((n_mc, len(popt))) @ raiz_cov.T
#     xc, fwhm, eta_muestra = draws[:,1], draws[:,2], draws[:,3]

#     df_cal = df_LaB6.sort_values('2theta Ka1 (°)')
#     x_cal = df_cal['2theta Ka1 (°)'].to_numpy()
#     beta_cal = df_cal['FWHM Ka1 (°)'].to_numpy()
#     err_beta_cal = df_cal['Error FWHM (°)'].to_numpy()
#     eta_cal = df_cal['eta'].to_numpy()
#     err_eta_cal = df_cal['Error eta'].to_numpy() if 'Error eta' in df_cal else np.zeros(len(df_cal))

#     # Conservar simulaciones dentro del intervalo angular calibrado
#     validos = (
#         (xc >= x_cal.min()) & (xc <= x_cal.max()) &
#         (fwhm > 0) & (eta_muestra >= 0) & (eta_muestra <= 1)
#     )
#     xc, fwhm, eta_muestra = xc[validos], fwhm[validos], eta_muestra[validos]

#     if len(xc) < 100:
#         return np.nan

#     # Interpolación de las contribuciones instrumentales y sus incertidumbres
#     idx = np.clip(np.searchsorted(x_cal, xc), 1, len(x_cal)-1)
#     x0, x1 = x_cal[idx-1], x_cal[idx]
#     w = (xc-x0)/(x1-x0)

#     beta_inst = (1-w)*beta_cal[idx-1] + w*beta_cal[idx]
#     err_beta_inst = np.sqrt(
#         ((1-w)*err_beta_cal[idx-1])**2 +
#         (w*err_beta_cal[idx])**2
#     )
#     beta_inst = rng.normal(beta_inst, err_beta_inst)

#     eta_inst = (1-w)*eta_cal[idx-1] + w*eta_cal[idx]
#     err_eta_inst = np.sqrt(
#         ((1-w)*err_eta_cal[idx-1])**2 +
#         (w*err_eta_cal[idx])**2
#     )
#     eta_inst = np.clip(rng.normal(eta_inst, err_eta_inst), 0, 1)
#     beta_inst = np.maximum(beta_inst, 0)

#     theta = np.deg2rad(xc/2)
#     beta_LaB6_intr = np.rad2deg(
#         K*(lambda_X/(D_LaB6*10))/np.cos(theta)
#     )

#     beta_G_inst = beta_inst*np.sqrt(1-eta_inst)
#     beta_L_inst = np.maximum(beta_inst*eta_inst-beta_LaB6_intr, 0)

#     beta_G_muestra = fwhm*np.sqrt(1-eta_muestra)
#     beta_L_muestra = fwhm*eta_muestra

#     beta_G_corr = np.sqrt(np.maximum(beta_G_muestra**2-beta_G_inst**2, 0))
#     beta_L_corr = np.maximum(beta_L_muestra-beta_L_inst, 0)

#     fwhm_corr = (
#         0.5346*beta_L_corr +
#         np.sqrt(0.2166*beta_L_corr**2+beta_G_corr**2)
#     )

#     fwhm_corr = fwhm_corr[np.isfinite(fwhm_corr)]
#     if len(fwhm_corr) < 100:
#         return np.nan

#     return np.std(fwhm_corr, ddof=1)
# #%% Gráficos de los ajustes del doblete
fig_LaB6c, axs = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
axs = axs.ravel()

for ax, (hkl, N, centro_ref) in zip(axs, reflexiones_LaB6):
    if hkl not in ajustes_LaB6:
        ax.set_visible(False)
        continue

    xp, yp, popt = ajustes_LaB6[hkl]
    xfit = np.linspace(xp.min(), xp.max(), 1000)
    yfit = doblete_pvoigt_fondo(xfit, *popt)

    ax.plot(xp, yp, '.', ms=2, color='0.35', label='Datos')
    ax.plot(xfit, yfit, '-', lw=1.3, label='Ajuste doblete')
    ax.axvline(popt[1], color='tab:red', ls='--', lw=0.8, label=r'$K\alpha_1$')
    theta2 = np.arcsin(np.clip(lambda_Ka2/lambda_Ka1*np.sin(np.deg2rad(popt[1]/2)), -1, 1))
    xc2 = np.rad2deg(2*theta2)
    ax.axvline(xc2, color='tab:blue', ls=':', lw=0.8, label=r'$K\alpha_2$')
    ax.set_title(hkl)
    ax.set_xlabel(r'$2\theta$ (°)')
    ax.grid(alpha=0.3)

axs[0].set_ylabel('Intensidad (u.a.)')
axs[4].set_ylabel('Intensidad (u.a.)')
axs[8].set_ylabel('Intensidad (u.a.)')
axs[0].legend(fontsize=8)
for ax in axs[len(reflexiones_LaB6):]:
    ax.set_visible(False)

plt.show()

#% Perfil instrumental: FWHM de Kalpha1 en función de 2theta
fig_Lab6_perfil, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
ax.errorbar(
    df_LaB6['2theta Ka1 (°)'],
    df_LaB6['FWHM Ka1 (°)'],
    yerr=df_LaB6['Error FWHM (°)'],
    fmt='o', capsize=3, label=r'$K\alpha_1$'
)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('FWHM (°)')
ax.set_title(r'Ancho del perfil de $K\alpha_1$ para LaB$_6$')
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
plt.show()

#% Resumen del parámetro de red
print(f'a promedio = {df_LaB6["a (Å)"].mean():.6f} Å')
print(f'Desviación estándar entre reflexiones = {df_LaB6["a (Å)"].std(ddof=1):.6f} Å')
print(f'Referencia NIST SRM 660c = {a_ref_LaB6:.6f} Å')
print(f'Diferencia relativa = {(df_LaB6["a (Å)"].mean()/a_ref_LaB6-1)*100:.4f} %')

#%% Importo Data
data_M17_raw = np.loadtxt('data/M17_crudo.txt', skiprows=1, delimiter='\t',
                          converters={0: lambda x: float(x.replace(',', '.')),
                                      1: lambda x: float(x.replace(',', '.'))})

data_M18_raw = np.loadtxt('data/M18_crudo.txt', skiprows=1,delimiter='\t',
                  converters={0: lambda x: float(x.replace(',', '.')),
                                      1: lambda x: float(x.replace(',', '.'))})
data_M19_raw = np.loadtxt('data/M19_crudo.txt', skiprows=1,delimiter='\t',
                  converters={0: lambda x: float(x.replace(',', '.')),
                                      1: lambda x: float(x.replace(',', '.'))})

data_M17_procesado1 = np.loadtxt('data/M17_procesado1.txt', skiprows=1,delimiter='\t',
                  converters={0: lambda x: float(x.replace(',', '.')),
                                      1: lambda x: float(x.replace(',', '.'))})
data_M17_procesado2 = np.loadtxt('data/M17_procesado2.txt', skiprows=1,delimiter='\t',
                  converters={0: lambda x: float(x.replace(',', '.')),
                                      1: lambda x: float(x.replace(',', '.'))})
data_M18_procesado = np.loadtxt('data/M18_procesado.txt', skiprows=1,delimiter='\t',
                  converters={0: lambda x: float(x.replace(',', '.')),
                                      1: lambda x: float(x.replace(',', '.'))})
data_M19_procesado = np.loadtxt('data/M19_procesado.txt', skiprows=1,delimiter='\t',
                  converters={0: lambda x: float(x.replace(',', '.')),
                                      1: lambda x: float(x.replace(',', '.'))})

#%% Parámetros para Análisis de Scherrer: pseudo-Voigt + fondo lineal
K = 0.9
lambda_X = 1.5406  # Å, Cu Kα
beta_inst_deg = None  # Sin patrón de referencia: no se corrige

# Posiciones aproximadas de las reflexiones de magnetita (2theta)
reflexiones = {
    '(220)': 30.1,
    '(311)': 35.5,
    '(400)': 43.1,
    '(422)': 53.4,
    '(511)': 57.0,
    '(440)': 62.6,
    '(620)': 71.0,
    '(533)': 74.0,
    '(444)': 79.0}
#%%
dostheta_M17_raw = data_M17_raw[:, 0]
intensity_M17_raw = data_M17_raw[:, 1]

dostheta_M18_raw = data_M18_raw[:, 0]
intensity_M18_raw = data_M18_raw[:, 1]

dostheta_M19_raw = data_M19_raw[:, 0]
intensity_M19_raw = data_M19_raw[:, 1]

dostheta_M17_procesado = data_M17_procesado1[:, 0]
intensity_M17_procesado = data_M17_procesado1[:, 1]

dostheta_M17_procesado2 = data_M17_procesado2[:, 0]
intensity_M17_procesado2 = data_M17_procesado2[:, 1]

dostheta_M18_procesado = data_M18_procesado[:, 0]
intensity_M18_procesado = data_M18_procesado[:, 1]

dostheta_M19_procesado = data_M19_procesado[:, 0]
intensity_M19_procesado = data_M19_procesado[:, 1]

labels=['M17 raw', 'M18 raw', 'M19 raw', 'M17 procesado', 'M17 procesado2', 'M18 procesado', 'M19 procesado']
colors=['C0', 'C1', 'C2', 'C3', 'C4']
#%% ploteo todos 
fig_all,(ax,ax2)=plt.subplots(2,1,figsize=(11,8), constrained_layout=True,sharex=True)
ax.plot(dostheta_M17_raw, intensity_M17_raw/max(intensity_M17_raw)+0.0,lw=0.9, label=labels[0],zorder=1)
ax.plot(dostheta_M18_raw, intensity_M18_raw/max(intensity_M18_raw)+0.2,lw=0.9, label=labels[1],zorder=2)
ax.plot(dostheta_M19_raw, intensity_M19_raw/max(intensity_M19_raw)+0.4,lw=0.9, label=labels[2],zorder=3)    


ax2.plot(dostheta_M17_procesado, intensity_M17_procesado/max(intensity_M17_procesado)+0.0,lw=0.9, label=labels[3],zorder=1)
ax2.plot(dostheta_M17_procesado2, intensity_M17_procesado2/max(intensity_M17_procesado2)+0.2,lw=0.9, label=labels[4],zorder=2)
ax2.plot(dostheta_M18_procesado, intensity_M18_procesado/max(intensity_M18_procesado)+0.4,lw=0.9, label=labels[5],zorder=3)    
ax2.plot(dostheta_M19_procesado, intensity_M19_procesado/max(intensity_M19_procesado)+0.6,lw=0.9, label=labels[6],zorder=1)
# ax2.plot(dostheta_M6, intensity_M6/max(intensity_M6)+0.6,lw=0.9, label=labels[7],zorder=2)

for hkl, posicion in reflexiones.items():
    ax.axvline(x=posicion, color='k', ls='--',alpha=0.8,lw=0.8)  
    ax2.axvline(x=posicion, color='k', ls='--',alpha=0.8,lw=0.8)
    ax2.text(posicion,1.05,hkl,rotation=0,ha='center',va='top',
            transform=ax2.get_xaxis_transform(),fontsize=9)

for a in (ax,ax2):
    a.legend(ncol=1,loc='upper left',frameon=True,shadow=True)
    # a.grid(axis='x')            
    a.set_ylabel('Intensidad (u.a.)')


ax.set_xlim(10,80)
ax2.set_xlabel('2θ (°)')
plt.suptitle('DRX en polvo')
plt.show()

#%% Normalización y recorte de difractogramas (25-40°)

datasets_raw = {
    'M17 raw': (dostheta_M17_raw, intensity_M17_raw),
    'M18 raw': (dostheta_M18_raw, intensity_M18_raw),
    'M19 raw': (dostheta_M19_raw, intensity_M19_raw),}

datasets_norm_raw = {}
for nombre, (x, y) in datasets_raw.items():
    mask = (x >= 28) & (x <= 40)
    datasets_norm_raw[nombre] = (x[mask], y[mask] / np.max(y))


datasets_procesado = {
'M17 procesado': (dostheta_M17_procesado, intensity_M17_procesado),
    'M17 procesado2': (dostheta_M17_procesado2, intensity_M17_procesado2),
    'M18 procesado': (dostheta_M18_procesado, intensity_M18_procesado),
    'M19 procesado': (dostheta_M19_procesado, intensity_M19_procesado)}
datasets_norm_procesado = {}
for nombre, (x, y) in datasets_procesado.items():
    mask = (x >= 28) & (x <= 40)
    datasets_norm_procesado[nombre] = (x[mask], y[mask] / np.max(y))


#%% Ploteo comparativo

fig, (ax,ax2) = plt.subplots(2,1,figsize=(11, 6), constrained_layout=True,sharex=True
                             )

for i, (nombre, (x, y)) in enumerate(datasets_norm_raw.items()):
    ax.plot(x, y + i*0.2,'.-', lw=1, label=nombre)

for i, (nombre, (x, y)) in enumerate(datasets_norm_procesado.items()):
    ax2.plot(x, y + i*0.2,'.-', lw=1, label=nombre)

ax.set_xlim(28, 40)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad normalizada (u.a.)')

ax2.set_xlabel('2θ (°)')
ax2.set_ylabel('Intensidad normalizada (u.a.)')

for a in (ax, ax2):
    a.legend(loc='upper left', frameon=True, shadow=True)
    a.grid(axis='x', alpha=0.3)
plt.show()
#%%
#%% Ajuste pseudo-Voigt de la reflexión (311) y cálculo de Scherrer

centro_ref = 35.5
ventana = 0.9

datasets_raw = {
    'M17 raw': (dostheta_M17_raw, intensity_M17_raw),
    'M18 raw': (dostheta_M18_raw, intensity_M18_raw),
    'M19 raw': (dostheta_M19_raw, intensity_M19_raw),
    }
resultados_311_raw = []
ajustes_311_raw = {}

datasets_procesado = {
'M17 procesado': (dostheta_M17_procesado, intensity_M17_procesado),
    'M17 procesado2': (dostheta_M17_procesado2, intensity_M17_procesado2),
    'M18 procesado': (dostheta_M18_procesado, intensity_M18_procesado),
    'M19 procesado': (dostheta_M19_procesado, intensity_M19_procesado)}

resultados_311_procesado = []
ajustes_311_procesado = {}

fig_raw, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
axs = axs.ravel()

for i, (nombre, (x, y)) in enumerate(datasets_raw.items()):
    try:
        # Ajuste pseudo-Voigt
        xp, popt, pcov, fwhm = analizar_pico(
            x, y, centro_ref,
            ventana=ventana,
            correcion_LaB6=False
        )

        A, xc, fwhm_deg, eta, b0, b1 = popt

        # Incertidumbres del ajuste
        errores = np.sqrt(np.maximum(np.diag(pcov), 0))
        err_xc = errores[1]
        err_fwhm = errores[2]

        # Cálculo de Scherrer
        theta_rad = np.deg2rad(xc/2)
        beta_rad = np.deg2rad(fwhm_deg)

        D = K*lambda_X/(beta_rad*np.cos(theta_rad))/10

        # Propagación de incertidumbre considerando
        # la covarianza entre xc y FWHM
        gradiente = np.array([
            D*np.tan(theta_rad)*np.pi/360,
            -D/fwhm_deg
        ])

        cov_xc_fwhm = pcov[np.ix_([1, 2], [1, 2])]
        err_D = np.sqrt(max(gradiente @ cov_xc_fwhm @ gradiente, 0))

        resultados_311_raw.append({
            'Muestra': nombre,
            '2theta (°)': xc,
            'Error centro (°)': err_xc,
            'FWHM (°)': fwhm_deg,
            'Error FWHM (°)': err_fwhm,
            'eta': eta,
            'D (nm)': D,
            'Error D (nm)': err_D
        })

        ajustes_311_raw[nombre] = (xp, popt, pcov)

        # Gráfico del ajuste
        mask = (x >= centro_ref-ventana) & (x <= centro_ref+ventana)
        yp = y[mask]

        xfit = np.linspace(xp.min(), xp.max(), 500)
        yfit = pvoigt_fondo(xfit, *popt)

        axs[i].plot(xp, yp, '.', ms=3, color='0.35', label='Datos')
        axs[i].plot(xfit, yfit, '-', lw=1.4, color='C1', label='Pseudo-Voigt')
        axs[i].axvline(xc, color='C3', ls='--', lw=0.8)

        axs[i].set_title(
            f'{nombre}\nFWHM = {fwhm_deg:.4f} ± {err_fwhm:.4f}°'
        )
        axs[i].set_xlabel(r'$2\theta$ (°)')
        axs[i].set_ylabel('Intensidad (u.a.)')
        axs[i].grid(alpha=0.3)

        print(
            f'{nombre}: D = {D:.2f} ± {err_D:.2f} nm'
        )

    except (RuntimeError, ValueError) as e:
        print(f'{nombre}: no se pudo ajustar ({e})')
        axs[i].set_visible(False)

for ax in axs[len(datasets_raw):]:
    ax.set_visible(False)

axs[0].legend(fontsize=8)
plt.suptitle('Ajuste pseudo-Voigt de la reflexión (311) de magnetita')
plt.show()



fig_procesado, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
axs = axs.ravel()

for i, (nombre, (x, y)) in enumerate(datasets_procesado.items()):
    try:
        # Ajuste pseudo-Voigt
        xp, popt, pcov, fwhm = analizar_pico(
            x, y, centro_ref,
            ventana=ventana,
            correcion_LaB6=False
        )

        A, xc, fwhm_deg, eta, b0, b1 = popt

        # Incertidumbres del ajuste
        errores = np.sqrt(np.maximum(np.diag(pcov), 0))
        err_xc = errores[1]
        err_fwhm = errores[2]

        # Cálculo de Scherrer
        theta_rad = np.deg2rad(xc/2)
        beta_rad = np.deg2rad(fwhm_deg)

        D = K*lambda_X/(beta_rad*np.cos(theta_rad))/10

        # Propagación de incertidumbre considerando
        # la covarianza entre xc y FWHM
        gradiente = np.array([
            D*np.tan(theta_rad)*np.pi/360,
            -D/fwhm_deg
        ])

        cov_xc_fwhm = pcov[np.ix_([1, 2], [1, 2])]
        err_D = np.sqrt(max(gradiente @ cov_xc_fwhm @ gradiente, 0))

        resultados_311_procesado.append({
            'Muestra': nombre,
            '2theta (°)': xc,
            'Error centro (°)': err_xc,
            'FWHM (°)': fwhm_deg,
            'Error FWHM (°)': err_fwhm,
            'eta': eta,
            'D (nm)': D,
            'Error D (nm)': err_D
        })

        ajustes_311_procesado[nombre] = (xp, popt, pcov)

        # Gráfico del ajuste
        mask = (x >= centro_ref-ventana) & (x <= centro_ref+ventana)
        yp = y[mask]

        xfit = np.linspace(xp.min(), xp.max(), 500)
        yfit = pvoigt_fondo(xfit, *popt)

        axs[i].plot(xp, yp, '.', ms=3, color='0.35', label='Datos')
        axs[i].plot(xfit, yfit, '-', lw=1.4, color='C1', label='Pseudo-Voigt')
        axs[i].axvline(xc, color='C3', ls='--', lw=0.8)

        axs[i].set_title(
            f'{nombre}\nFWHM = {fwhm_deg:.4f} ± {err_fwhm:.4f}°'
        )
        axs[i].set_xlabel(r'$2\theta$ (°)')
        axs[i].set_ylabel('Intensidad (u.a.)')
        axs[i].grid(alpha=0.3)

        print(
            f'{nombre}: D = {D:.2f} ± {err_D:.2f} nm'
        )

    except (RuntimeError, ValueError) as e:
        print(f'{nombre}: no se pudo ajustar ({e})')
        axs[i].set_visible(False)

for ax in axs[len(datasets_procesado):]:
    ax.set_visible(False)

axs[0].legend(fontsize=8)
plt.suptitle('Ajuste pseudo-Voigt de la reflexión (311) de magnetita')
plt.show()
#%% Tabla de resultados

tabla_311 = pd.DataFrame(resultados_311_raw)
display(tabla_311.round(5))


tabla_311_procesado = pd.DataFrame(resultados_311_procesado)
display(tabla_311_procesado.round(5))
#%% Salvo figuras

#%% Salvo figuras

carpeta = 'figuras_DRX'
os.makedirs(carpeta, exist_ok=True)

figuras = {
    'DRX_difractogramas_comparacion': fig_all,
    'DRX_difractogramas_normalizados': fig,
    'DRX_ajustes_311_raw': fig_raw,
    'DRX_ajustes_311_procesado': fig_procesado
}

for nombre, figura in figuras.items():
    figura.savefig(
        os.path.join(carpeta, f'{nombre}.png'),
        dpi=300, bbox_inches='tight', facecolor='white'
    )

print(f'Se guardaron {len(figuras)} figuras en "{carpeta}/"')

#%%