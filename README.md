# **MACA**: **Ma**sk and **Ca**ption what you see

This is 2 implementations of MACA on python3, keras and tensorflow2. The model generates pixel-level masks for each instance of an object in the image and bounding-box-level captions for each area with meaningful contents. It is based on Mask R-CNN and DenseCap.

![](https://drive.google.com/file/d/1LGCMVXPpauwwi9WTsI1osnwp_xJ8adtv/view)

The repository includes:


*   Jupyter notebook of using MACA to detect single images and videos
*   Jupyter notebook of training MACA further on CocoVG dataset
*   Evaluation metrics on CocoVG dataset (mAP, BLUE, CIDEr, ROUGE)
*   Requirements file of building programming environments


The basic theory source code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below).



# **Getting Started**

 

*   **maca_detecting.ipynb** is a good start for user to detect images and videos directly, this notebook contains the way to download necessary materials and how to use them to detect on single images or videos. This note book also contains tutorial about how to calculate the evaluation metrics of MACA on CocoVG validation & and training dataset.
*   **maca_training.ipynb** is a tutorial about how to download necessary materials and implement them to further train MACA on CocoVG dataset.
*   **assets** directory contains the images used in the dissertation.


### **Necessary materials include:**

*   **Source code of MACA**
*   **Well-trained model weights** (maca_cocovg.hdf5)
*   **Data of CocoVG dataset**

***(All of the necessary materials will be downloaded automatically by the code in jupyter notebook)***


**Note:** Running online in google colab is recommended (you do not need to set up any coding environments), you can check this [link](https://colab.research.google.com/notebooks/intro.ipynb) to learn how to run jupyter notebook on google colab. Also you can run all of the jupyter notebooks locally on your device, you need to make sure to successfully setup python3 + jupyter notebook environments on your device (also cudnn + Nvidia environments if you want to use GPU to accelerate), the tutorial of setting up environment can be seen [here](https://www.codecademy.com/articles/install-python3).


# **Other Informations**

If you are interested in background basic theories and the way to code them, please check my personal github repository [here](https://github.com/Askfk/maca).


If you are interested in well-trained weights or CocoVG dataset, please check the release [here](https://github.com/Askfk/maca/releases)



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