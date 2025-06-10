# blinkEyeDetection

Göz kırpma algılama ve sayma uygulaması.

Yüz tanıma ve haritalandırmak için gerekli olan `shape_predictor_68_face_landmarks.dat`
dosyasını [Google Drive](https://drive.google.com/drive/folders/1s40odDZNdRgJtNrxwbWlUGsu6d89eF31?usp=sharing)
üzerinden indirebilirsiniz.

## Kullanım

```
python blink.py [-p PATH_TO_PREDICTOR] [-v VIDEO]
                [--eye-ar-thresh T] [--eye-ar-consec N]
                [--drowsy-frames N] [--mar-thresh T]
                [--yawn-consec N]
```

`-v` parametresi verilmezse varsayılan olarak webcam kullanılır.

Ek parametreler sayesinde göz kırpma ve esneme eşikleri değiştirilebilir.
Uzun süreli göz kapanması veya ardışık esneme durumlarında yorgunluk uyarısı
verilir.
