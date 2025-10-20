# Histology Image Segmentation Using Pretrained Vision Transformers

This is the official GitHub repository for the paper:

> **Histology Image Segmentation Using Pretrained Vision Transformers**

---

## 📄 Abstract

Developing robust segmentation methods that can adapt to new datasets is a critical research challenge in histology image processing. Histology images vary significantly due to factors like staining and tissue quality, causing models trained on one dataset to perform poorly on another. Consequently, there is a growing interest in developing models that leverage large-scale pretraining on diverse datasets to learn generalized representations, enabling better adaptation to specific downstream tasks.

In this paper, we propose an automated segmentation method based on a pretrained Vision Transformer for two specific tasks: epithelium segmentation in testis tissue and gland segmentation in colorectal tissue. In the suggested method, the pretrained Vision Transformer is used as an encoder, while a decoder is trained on the target training set. We evaluated two pretrained encoders: the pretrained Vision Transformer from **SAM**, which is trained on natural images, and the pretrained Vision Transformer from **UNI**, which is explicitly trained on histology images.  

Based on our experiments, for these two specific tasks, the **SAM encoder**, despite being trained on natural images, achieved better segmentation results than **UNI**, the domain-specific encoder. The proposed method outperforms a baseline CNN model in both tasks, with an accuracy of **88.8%** for the testis tissue and **85.05%** for colorectal tissue.  

To evaluate the suggested method's generalization capability, we tested it on histology images from different animal species and with staining protocols that differ from those in the training set. The proposed method shows significantly better performance compared to the baseline model.

---

## 🧠 Method Overview

Below is the flowchart illustrating the overall architecture of our proposed method.

![Model Flowchart](Fakhr1_flow.pdf)

---

## 👩‍💻 Authors

- **Azadeh Fakhrzadeh**<sup>1,+,*</sup>  
- **Alborz Esfandyari**<sup>1,2,+</sup>  
- **Amir Hossein Seddighi**<sup>1,+</sup>  
- **Lena Holm**<sup>3</sup>  
- **Christian Sonne**<sup>4</sup>  
- **Rune Dietz**<sup>4</sup>  
- **Ellinor Spörndly-Nees**<sup>5</sup>

<sup>1</sup> *Iranian Research Institute for Information Science and Technology (IranDoc), Information Technology Department, Tehran, Iran*  
<sup>2</sup> *Isfahan University of Technology, Department of Electrical and Computer Engineering, Isfahan, Iran*  
<sup>3</sup> *Swedish University of Agricultural Sciences, Department of Animal Biosciences, Uppsala, Sweden*  
<sup>4</sup> *Aarhus University, Department of Ecoscience, Arctic Research Centre, Roskilde, Denmark*  
<sup>5</sup> *National Veterinary Institute, Department of Pathology and Wildlife Diseases, Uppsala, Sweden*  

<sup>*</sup> Corresponding author: **fakhrzadeh@irandoc.ac.ir**  
<sup>+</sup> These authors contributed equally to this work.

---

## 💰 Funding

This work was supported by the **Danish Cooperation for Environment in the Arctic (DANCEA)**, **The Commission for Scientific Research in Greenland (KVUG)**, **The Prince Albert II Foundation**, and the **Arctic Research Centre (ARC)** at Aarhus University.  
The study was also part of the **International Polar Year (IPY) BearHealth project** (IPY 2007–2008, Activity #134) funded by the **Independent Research Fund Denmark**.

---

## 🙏 Acknowledgements

We are grateful for the excellent technical assistance of **Gunilla Ericson-Forslund** and **Astrid Gumucio** in histological preparations.  
We also acknowledge the subsistence hunters in Scoresby Sound for obtaining samples.

---

## ✍️ Author Contributions

A.F., A.E., and A.H.S. conceived the study, developed the methodology and model, conducted the experiments, performed the analysis, and prepared the manuscript.  
L.H., C.S., R.D., and E.S.-N. prepared the dataset, provided background materials and domain expertise, and critically reviewed the manuscript.  
All authors reviewed and approved the final version of the manuscript.  
No conflict of interest was reported.

---

## 📚 Citation

If you find our work useful for your research, please consider citing it.  
Your citation is greatly appreciated and helps support our research efforts.

```
@article{fakhrzadeh2025histologyvit,
  title={Histology Image Segmentation Using Pretrained Vision Transformers},
  author={Fakhrzadeh, Azadeh and Esfandyari, Alborz and Seddighi, Amir Hossein and Holm, Lena and Sonne, Christian and Dietz, Rune and Spörndly-Nees, Ellinor},
  year={2025},
  note={Under Review}
}
```

---

## 📬 Contact

For questions or collaboration inquiries, please contact:  
**fakhrzadeh@irandoc.ac.ir**

---

⭐ **If this repository or paper helps your research, please give it a star — your support means a lot!**
