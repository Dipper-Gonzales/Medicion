"""
Calibración de cámara (corrige la distorsión de lente) con tablero de ajedrez.

Uso:
    python calibrar_camara.py --fuente http://192.168.1.50:4747/video
    python calibrar_camara.py --fuente 1            # índice de cámara (DroidCam USB/cliente PC)

Teclas:
    ESPACIO  guardar el frame actual (solo si se detecta el tablero)
    c        calcular la calibración (necesita al menos --min capturas)
    q        salir

Consejos:
    - Imprime un tablero de ajedrez y pégalo sobre algo rígido y plano.
    - --cols y --filas son las ESQUINAS INTERNAS (un tablero de 10x7 cuadros tiene 9x6).
    - Toma 15-25 capturas con el tablero en distintas posiciones, inclinaciones y
      cerca de los bordes de la imagen.
    - Usa la MISMA resolución que usarás luego para medir y con el enfoque bloqueado.
"""
import argparse
import sys

import cv2
import numpy as np


def abrir_camara(fuente):
    if str(fuente).isdigit():
        fuente = int(fuente)
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        print(f"No se pudo abrir la fuente: {fuente}")
        sys.exit(1)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente", default="0", help="índice de cámara o URL de DroidCam")
    ap.add_argument("--cols", type=int, default=9, help="esquinas internas en horizontal")
    ap.add_argument("--filas", type=int, default=6, help="esquinas internas en vertical")
    ap.add_argument("--cuadro", type=float, default=2.5, help="lado de un cuadro en cm")
    ap.add_argument("--min", type=int, default=15, help="mínimo de capturas")
    ap.add_argument("--salida", default="calib.npz")
    args = ap.parse_args()

    patron = (args.cols, args.filas)
    objp = np.zeros((args.cols * args.filas, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.filas].T.reshape(-1, 2) * args.cuadro

    puntos_obj, puntos_img = [], []
    tam = None
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cap = abrir_camara(args.fuente)
    print("ESPACIO = guardar captura | c = calibrar | q = salir")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No llega video de la cámara.")
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tam = gris.shape[::-1]  # (ancho, alto)
        encontrado, esquinas = cv2.findChessboardCorners(
            gris, patron,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        vista = frame.copy()
        if encontrado:
            esquinas = cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), criterio)
            cv2.drawChessboardCorners(vista, patron, esquinas, encontrado)

        estado = "TABLERO OK" if encontrado else "buscando tablero..."
        cv2.putText(vista, f"{estado} | capturas: {len(puntos_img)}/{args.min}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if encontrado else (0, 0, 255), 2)
        cv2.imshow("Calibracion", vista)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" ") and encontrado:
            puntos_obj.append(objp)
            puntos_img.append(esquinas)
            print(f"Captura {len(puntos_img)} guardada")
        elif k == ord("c"):
            if len(puntos_img) < args.min:
                print(f"Faltan capturas: {len(puntos_img)}/{args.min}")
                continue
            rms, K, dist, _, _ = cv2.calibrateCamera(puntos_obj, puntos_img, tam, None, None)
            np.savez(args.salida, K=K, dist=dist, tam=np.array(tam))
            print(f"\nError RMS de reproyección: {rms:.3f} px  (menor a ~0.5 es bueno)")
            print(f"Matriz de cámara:\n{K}")
            print(f"Distorsión: {dist.ravel()}")
            print(f"Guardado en {args.salida} (resolución {tam[0]}x{tam[1]})")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()