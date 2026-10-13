"""
Medición de objetos con cámara (OpenCV) usando un marcador ArUco como referencia de escala.
 
Uso básico:
    python medir.py --fuente http://192.168.1.50:4747/video --marcador-cm 5.0
 
Con calibración de lente (recomendado):
    python medir.py --fuente 1 --marcador-cm 5.0 --calib calib.npz
 
Con objetos que tienen altura (corrige la escala de su cara superior):
    python medir.py --fuente 1 --marcador-cm 5.0 --calib calib.npz --altura-cm 2.5
 
Sin marcador (escala manual, solo si la cámara NO se mueve):
    python medir.py --fuente 1 --cm-por-px 0.035
 
Requisitos:
    pip install "opencv-python>=4.7" numpy
 
Cómo preparar la escena:
    1. Genera e imprime un marcador ArUco (diccionario 4x4_50, id 0). Mide con regla el
       lado del cuadro NEGRO exterior y pásalo en --marcador-cm.
       Para generarlo:  python medir.py --generar-marcador
    2. Cámara fija, mirando perpendicular a la superficie (vista cenital).
    3. El marcador y los objetos deben estar sobre el MISMO plano.
    4. Fondo uniforme con buen contraste respecto al objeto, luz pareja, sin sombras duras.
    5. Bloquea enfoque, exposición y balance de blancos en DroidCam; no uses zoom.
 
Teclas:
    q  salir
    s  guardar captura (captura_N.png)
    l  bloquear / desbloquear la escala actual (ignora el marcador si lo tapas)
    t  cambiar entre objetos oscuros sobre fondo claro y al revés
    m  mostrar / ocultar la máscara binaria
    +  subir 0.5 cm la altura del objeto (necesita --calib)
    -  bajar 0.5 cm la altura del objeto
"""
import argparse
import sys
from collections import deque
 
import cv2
import numpy as np
 
ID_MARCADOR = 0
DICCIONARIO = cv2.aruco.DICT_4X4_50
 
 
def abrir_camara(fuente):
    if str(fuente).isdigit():
        fuente = int(fuente)
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"No se pudo abrir la fuente: {fuente}")
        sys.exit(1)
    return cap
 
 
def generar_marcador():
    dic = cv2.aruco.getPredefinedDictionary(DICCIONARIO)
    img = cv2.aruco.generateImageMarker(dic, ID_MARCADOR, 600)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite("marcador_aruco.png", img)
    print("Guardado marcador_aruco.png. Imprímelo SIN escalar ('tamaño real') "
          "y mide con regla el lado del cuadrado negro.")
 
 
def cargar_calibracion(ruta, tam_frame):
    datos = np.load(ruta)
    K, dist, tam = datos["K"], datos["dist"], tuple(datos["tam"])
    if tam != tam_frame:
        print(f"AVISO: la calibración es de {tam[0]}x{tam[1]} y el video es "
              f"{tam_frame[0]}x{tam_frame[1]}. Se ignora la corrección de lente; "
              "recalibra con esta resolución.")
        return None
    nueva_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, tam_frame, 0, tam_frame)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, nueva_K, tam_frame, cv2.CV_32FC1)
    # fx de la imagen YA corregida: sirve para estimar la distancia de la cámara a la mesa
    return mapx, mapy, float(nueva_K[0, 0])
 
 
def escala_desde_marcador(detector, gris, marcador_cm):
    """Devuelve (cm_por_px, esquinas_del_marcador) o (None, None)."""
    esquinas, ids, _ = detector.detectMarkers(gris)
    if ids is None:
        return None, None
    for c, i in zip(esquinas, ids.ravel()):
        if i == ID_MARCADOR:
            pts = c.reshape(4, 2)
            lados = [np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]
            return marcador_cm / float(np.mean(lados)), pts
    return None, None
 
 
def segmentar(gris, oscuro_sobre_claro, esquinas_marcador):
    desenfoque = cv2.GaussianBlur(gris, (7, 7), 0)
    tipo = cv2.THRESH_BINARY_INV if oscuro_sobre_claro else cv2.THRESH_BINARY
    _, binaria = cv2.threshold(desenfoque, 0, 255, tipo + cv2.THRESH_OTSU)
 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
 
    # Borra el marcador para que no se mida como si fuera un objeto
    if esquinas_marcador is not None:
        mascara = np.zeros_like(binaria)
        cv2.fillConvexPoly(mascara, esquinas_marcador.astype(np.int32), 255)
        mascara = cv2.dilate(mascara, np.ones((25, 25), np.uint8))
        binaria[mascara > 0] = 0
    return binaria
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="0", help="índice de cámara o URL de DroidCam")
    ap.add_argument("--marcador-cm", type=float, help="lado del marcador ArUco en cm")
    ap.add_argument("--cm-por-px", type=float, help="escala manual si no usas marcador")
    ap.add_argument("--calib", help="archivo .npz generado por calibrar_camara.py")
    ap.add_argument("--area-min", type=int, default=1500, help="área mínima del contorno (px²)")
    ap.add_argument("--claro", action="store_true",
                    help="objetos CLAROS sobre fondo oscuro (por defecto: oscuros sobre claro)")
    ap.add_argument("--altura-cm", type=float, default=0.0,
                    help="altura de los objetos sobre la mesa (donde está el marcador); "
                         "corrige la escala de su cara superior. Requiere --calib")
    ap.add_argument("--generar-marcador", action="store_true")
    args = ap.parse_args()
 
    if args.generar_marcador:
        generar_marcador()
        return
 
    if args.marcador_cm is None and args.cm_por_px is None:
        print("Indica --marcador-cm (recomendado) o --cm-por-px.")
        sys.exit(1)
 
    cap = abrir_camara(args.fuente)
    ok, frame = cap.read()
    if not ok:
        print("No llega video de la cámara.")
        sys.exit(1)
    alto, ancho = frame.shape[:2]
    mapas = cargar_calibracion(args.calib, (ancho, alto)) if args.calib else None
    fx = mapas[2] if mapas is not None else None
    altura = args.altura_cm
    if altura > 0 and fx is None:
        print("AVISO: --altura-cm necesita una --calib válida (para conocer la focal). Se ignora.")
        altura = 0.0
 
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICCIONARIO), cv2.aruco.DetectorParameters()
    )
 
    historial = deque(maxlen=20)  # suaviza la escala con la mediana de los últimos frames
    escala_bloqueada = None
    frames_sin_marcador = 0
    oscuro = not args.claro
    ver_mascara = False
    n_captura = 0
 
    print("Teclas: q salir | s captura | l bloquear escala | t invertir | m máscara")
 
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Se perdió el video.")
            break
        if mapas is not None:
            frame = cv2.remap(frame, mapas[0], mapas[1], cv2.INTER_LINEAR)
 
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # --- Escala (cm por píxel) ---
        esquinas = None
        if args.marcador_cm is not None:
            esc, esquinas = escala_desde_marcador(detector, gris, args.marcador_cm)
            frames_sin_marcador = 0 if esc is not None else frames_sin_marcador + 1
            if esc is not None and escala_bloqueada is None:
                historial.append(esc)
            if esquinas is not None:
                cv2.polylines(frame, [esquinas.astype(np.int32)], True, (255, 0, 0), 2)
            cm_px = escala_bloqueada or (float(np.median(historial)) if historial else None)
        else:
            cm_px = args.cm_por_px
 
        if cm_px is None:
            cv2.putText(frame, "No veo el marcador ArUco", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("Medicion", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
 
        # --- Corrección por altura: la cara superior está más cerca de la cámara ---
        # Modelo pinhole: escala a altura h = escala del plano - h / f
        cm_px_obj = cm_px - altura / fx if (altura > 0 and fx) else cm_px
 
        # --- Segmentación y medición ---
        binaria = segmentar(gris, oscuro, esquinas)
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        for c in contornos:
            if cv2.contourArea(c) < args.area_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + w >= ancho - 2 or y + h >= alto - 2:
                continue  # el objeto toca el borde: la medición saldría cortada
 
            rect = cv2.minAreaRect(c)  # funciona con objetos rotados
            (cx, cy), (rw, rh), _ = rect
            largo, anch = sorted((rw * cm_px_obj, rh * cm_px_obj), reverse=True)
 
            caja = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(frame, [caja], 0, (0, 255, 0), 2)
            etiqueta = f"{largo:.1f} x {anch:.1f} cm"
            cv2.putText(frame, etiqueta, (int(cx) - 70, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(frame, etiqueta, (int(cx) - 70, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
 
        info = f"escala: {cm_px:.5f} cm/px" + (" [BLOQUEADA]" if escala_bloqueada else "")
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        linea2 = []
        if fx:
            linea2.append(f"camara a ~{cm_px * fx:.0f} cm de la mesa")
        if altura > 0:
            linea2.append(f"altura objeto: {altura:.1f} cm")
        if linea2:
            cv2.putText(frame, " | ".join(linea2), (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if args.marcador_cm is not None and not escala_bloqueada and frames_sin_marcador > 15:
            cv2.putText(frame, "MARCADOR NO VISIBLE: escala desactualizada", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow("Medicion", frame)
        if ver_mascara:
            cv2.imshow("Mascara", binaria)
 
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            n_captura += 1
            cv2.imwrite(f"captura_{n_captura}.png", frame)
            print(f"Guardada captura_{n_captura}.png")
        elif k == ord("l"):
            escala_bloqueada = None if escala_bloqueada else cm_px
        elif k == ord("t"):
            oscuro = not oscuro
        elif k == ord("m"):
            ver_mascara = not ver_mascara
            if not ver_mascara:
                cv2.destroyWindow("Mascara")
        elif k in (ord("+"), ord("=")) and fx:
            altura += 0.5
        elif k in (ord("-"), ord("_")) and fx:
            altura = max(0.0, altura - 0.5)
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()