# Low-Light-Image-Enhancement-using-GAN
In this project, text is extracted from an image and is converted into text format. It works on both typed and hadwritten text, but only works where text is in Capital Letters.

<h4>Dataset Used:</h4>Extended MNIST (Letters) :- emnist-byclass-train.csv, emnist-byclass-test.csv

<h4>Kaggle Link for Dataset:</h4>https://www.kaggle.com/crawford/emnist

<h4>Link to Published Paper:</h4>https://www.irjet.net/archives/V7/i5/IRJET-V7I5964.pdf

<h4>System Configuration</h4>
<ul>
  <li>Processor: Intel i7 9th gen</li>
  <li>GPU: Nvidia RTX 2060 Mobile</li>
  <li>RAM: 16GB</li>
</ul>

<h4>Softwares used for this project</h4>
<ol>
  <li>Python 3.7</li>
  <li>Tensorflow 2.2.1</li>
</ol>

<h3>Steps to run the project</h3>
  <b>**Change the File paths in the codes to their respective paths before execution.**</b>
<ol>
  <li>Run csv_to_image(emnist).py to convert from csv to image format.</li>
  <li>Run letters(emnist).py to train the CNN model on EMNIST dataset.</li>
  <li>Run bounding box.py to predict to extract text from image.</li>
</ol>

<h3>CNN Architecture used</h3>
<img src="https://github.com/DragonSinMeliodas99/Optical-Character-Recognition/blob/main/Layers.png" alt="CNN layers" width="900"></img>
