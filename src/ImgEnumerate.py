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


