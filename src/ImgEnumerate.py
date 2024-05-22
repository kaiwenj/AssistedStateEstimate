import numpy as np

def ImgEnumerate(OriginalImg,KImg):
    MaskIdx=np.where(np.all(np.isnan(KImg),axis=1))[0]
    NumMasked=len(MaskIdx)
    OutImg=np.empty((NumMasked,OriginalImg.shape[0],OriginalImg.shape[1]))
    for i in range(NumMasked):
        TempImg=KImg.copy()
        TempImg[MaskIdx[i],:]=OriginalImg[MaskIdx[i],:]
        OutImg[i]=TempImg
    return OutImg


c = np.array([[1, 2, 1],
                  [2, 2, 2],
                  [7, 9, 3]], dtype=np.float32)

c[1,:]=np.nan


d=np.array([[1, 2, 1],
                  [2, 2, 2],
                  [7, 9, 3]], dtype=np.float32)


ImgEnumerate(d,c)

np.where(np.all(np.isnan(c),axis=1))[0][1]

np.array(list(range(0,3)))[np.all(np.isnan(c),axis=1)]


for i in range(4):
    print(i)