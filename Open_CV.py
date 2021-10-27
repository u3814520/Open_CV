import cv2
import dlib

cap = cv2.VideoCapture('./ccc.mp4')
FPS = cap.get(cv2.CAP_PROP_FPS)  # frame per second
F_Count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # frame count
print(f'FPS : {FPS:.2f} ms, Frame_Count : {F_Count}')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 取得畫面尺寸
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector = dlib.get_frontal_face_detector()  # Dlib 的人臉偵測器

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=150, nmixtures=5, backgroundRatio=0.7)  # try history = 1
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # XVID MJPG
out = cv2.VideoWriter('./video/hw.mp4', 0x00000021, 23.98, (width, height))

while (1):  # MOG
    ret, frame = cap.read()
    if not ret:
        break
    if ret:
        r2 = fgbg.apply(frame)  # fgbg
        text = 'MOG'
        # text2 = 'Merry Christmas!'
        # cv2.putText(r2, text2, (100, 600),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(r2, text, (100, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow('hw', r2)
        out.write(r2)
        k = cv2.waitKey(100) & 0xff
        if 1840 <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) <= 1890:
            break
        if k == 27:
            break
    else:
        break

while (1):  # Canny
    ret, frame = cap.read()
    if not ret:
        break
    if ret:
        r2 = cv2.Canny(frame, 32, 80)  # different threshold
        text = 'Canny'
        text2 = 'Merry Christmas!'
        cv2.putText(r2, text2, (100, 600), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(r2, text, (100, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 4)
        cv2.imshow('hw', r2)
        out.write(r2)
        k = cv2.waitKey(100) & 0xff
        if 7950 <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < 8050:
            break
        if k == 27:
            break
    else:
        break

while (1):  # 人臉辨識
    ret, r2 = cap.read()
    text = 'face'
    text2 = 'Merry Christmas!'
    cv2.putText(r2, text2, (100, 600), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(r2, text, (100, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 4, cv2.LINE_AA)
    if not ret:
        break
    if ret:
        face_rects, scores, idx = detector.run(r2, 0, -.3)  # 偵測人臉
        for i, d in enumerate(face_rects):  # 取出所有偵測的結果
            x1 = d.left();
            y1 = d.top();
            x2 = d.right();
            y2 = d.bottom()
            text = f'{scores[i]:.2f}, ({idx[i]:0.0f})'

            cv2.rectangle(r2, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)  # 以方框標示偵測的人臉
            cv2.putText(r2, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)  # 標示分數
        cv2.imshow('hw', r2)  # 顯示結果
        out.write(r2)
        k = cv2.waitKey(80) & 0xff
        if 10950 <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < 11100:
            break
        if k == 27:
            break
    else:
        break

while (1):  #Sobel
    ret, r2 = cap.read()
    if not ret:
        break
    if ret:
        text = 'Sobel'
        text2 = 'Merry Christmas!'
        cv2.putText(r2, text2, (100, 600), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(r2, text, (100, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 255, 255), 4, cv2.LINE_AA)
        x_grad = cv2.Sobel(r2,cv2.CV_32F, 1, 0)
        y_grad = cv2.Sobel(r2,cv2.CV_32F, 0, 1)
        x_grad = cv2.convertScaleAbs(x_grad)
        y_grad = cv2.convertScaleAbs(y_grad)
        r2 = cv2.add(x_grad, y_grad, dtype=cv2.CV_16S)
        r2 = cv2.convertScaleAbs(r2)
        cv2.imshow('hw',r2)
        out.write(r2)
        k= cv2.waitKey(80) & 0xff
    if 15000 <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < 15100:
        break
    if k == 27:
        break


while (1): # GRAY
    ret, frame = cap.read()
    r2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:
        text = 'GRAY'
        text2 = 'Merry Christmas!'
        cv2.putText(r2, text2, (100, 600), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(r2, text, (100, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.imshow('hw', r2)
        out.write(r2)
        k= cv2.waitKey(80) & 0xff
        if 20000 <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < 20150:
            break
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
