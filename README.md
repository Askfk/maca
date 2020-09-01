# **MACA**: **Ma**sk and **Ca**ption what you see  (UoB Summer Project)

This is 2 implementations of MACA on python3, keras and tensorflow2. The model generates pixel-level masks for each instance of an object in the image and bounding-box-level captions for each area with meaningful contents. It is based on Mask R-CNN and DenseCap.

![](https://github.com/Askfk/maca/blob/master/readme/1.png)

The repository includes:


*   Jupyter notebook of using MACA to detect single images and videos
*   Jupyter notebook of training MACA further on CocoVG dataset
*   Evaluation metrics on CocoVG dataset (mAP, BLUE, CIDEr, ROUGE)
*   Requirements file of building programming environments


The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below).



# **Citation**


```
@misc{Askfk_maca_2020,
  title={MACA: Mask and Caption what you see},
  author={Yiming Li},
  year={2020},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/Askfk/maca}},
}
```

