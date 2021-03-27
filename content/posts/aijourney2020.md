---
title: Historical Manuscripts Recongition using AttentionOCR — AI Journey 2020
date: "2020-11-26T23:46:37.121Z"
template: "post"
draft: false
slug: "aijourney2020"
category: "Competitions"
tags:
  - "Contests solutions"
  - "Computer Vision"
  - "OCR"
description: "In this article we want to share our team’s 2nd place solution for Artificial Intelligence Journey Junior 2020 Competition (Digital Peter track). This contest was about line-by-line recognition of Peter the Great’s manuscripts. The task is related to several AI technologies (Computer Vision, NLP, and knowledge graphs)."
socialImage: "/media/image-2.jpg"
---

<b>OCR (Optical Character Recognition)</b> is a quite common task in computer vision. There are plenty of methods, approaches and tools for solving this kind of problems. However, when it comes to working with unusual data like manuscripts or ancient records, you have to implement your own unique approach since there are no open-source packages for this particular task.
<br><br>
In this article we want to share our team’s <b>2nd place solution for Artificial Intelligence Journey Junior 2020 Competition (Digital Peter track)</b>. This contest was about line-by-line recognition of Peter the Great’s manuscripts. The task is related to several AI technologies (Computer Vision, NLP, and knowledge graphs). Competition data was prepared by Sberbank of Russia, Saint Petersburg Institute of History (N.P.Lihachov mansion) of Russian Academy of Sciences, Federal Archival Agency of Russia and Russian State Archive of Ancient Acts.

Our team:
* Maksim Zhdanov: [GitHub](https://github.com/xzcodes), [LinkedIn](https://www.linkedin.com/in/maksim-zhdanov-2a2b7819a/), [Kaggle](https://www.kaggle.com/dwdkills)
* Vadim Timakin: [GitHub](https://github.com/t0efL), [LinkedIn](https://www.linkedin.com/in/vadim-timakin-6298b91b6/), [Kaggle](https://www.kaggle.com/vadimtimakin)

### Data description

So, let’s breakdown the data. We had more than 6000 images with cut lines from the manuscripts. There could be single characters, words or even very long sentences, so image size differed for each picture. Let’s look at the example of an image:

![Regular sample that contains a whole sentence](https://miro.medium.com/max/875/0*nDIU-hI0gTjs0pkb.jpg)

Kind of bad handwriting, huh? In fact, AI can achieve quite low character error rate (CER) in this task. The label for that sample:
<br>
<b>зело многа в гафѣ i непърестано выхо</b>

Also, there were vertical rotated samples in the training set, which influenced error a lot, so that had to be fixed. Moreover, training data contained loads of images with crossed text.

![Short sample with crossed text](https://miro.medium.com/max/469/0*Y8YVh1hlntYQXUQA.png)

### Data processing

In this section you can learn about useful ideas in data processing that boosted our score.
* <b>Transforming images to grayscale.</b> Pretty obvious that inputing handwriting text in grayscale helps as it becomes easier to learn important features from text.
* <b>Tiny image rotation.</b> We rotated our samples in range from -2 to 2 degrees to expand the training set. That helped a lot.
* <b>Downscaling images.</b> We noticed that after applying rotation some small letters could erase, so we downscaled images for a bit to ensure that everything is ok.
<br>

![Downscaled images. Default background is white, black was chosen just for transparency](https://miro.medium.com/max/633/0*pJEIQu5n9-xJgAw_)

* <b>Smart resizing.</b> Implementation of this approach is especially interesting. Since we had various image sizes, samples those contained only one letter were highly stretched out because of the resizing parameters (width = 1024, height = 128). It was crucial to save aspect ratios of an images. The solution was a concatenation of original image with a white area on different sides to transform image’s aspect ratio to 8:1 (1024:128).
<br>

![That’s how this method works](https://miro.medium.com/max/875/0*3knr-JZnxPFqj8IV.png)

~~~~python
import numpy as np
from PIL import Image
import cv2
import PIL

class SmartResize:
    """
    Resizes image avoiding changing its aspect ratio.
    """
 
    def __init__(self, width, height, fillcolor=255):
        """
        Args:
        
            width (int): target width of the image.
            
            height (int): target height of the image.
            
            fillcolor (int): defults to 255 - white. Number in range [0, 255]
            representing fillcolor.
        """
        assert 0 <= fillcolor <= 255, "fillcolor has to contain values in " \
                                      "range [0, 255]. "
 
        self.width = int(width)
        self.height = int(height)
        self.ratio = int(width / height)
        self.color = fillcolor
 
    def __call__(self, img) -> PIL.Image.Image:
        """
        Transformation.
        Args:
            img (PIL.Image.Image): RGB PIL image which has to be transformed.
        """
        img = np.array(img)
        h, w, _ = img.shape
 
        if not (w / h) == self.ratio:
            if (w / h) < self.ratio:
                white = np.zeros([h, self.ratio * h - w, 3], dtype=np.uint8)
                white.fill(self.color)
                img = cv2.hconcat([img, white])
            elif (w / h) > self.ratio:
                white = np.zeros(
                    [(w - self.ratio * h) // (self.ratio * 2), w, 3],
                    dtype=np.uint8)
                white.fill(self.color)
                img = cv2.vconcat([white, img, white])
        img = cv2.resize(img, (self.width, self.height))
 
        img = Image.fromarray(img.astype(np.uint8))
        return img
 
    def __repr__(self) -> str:
        """Representation."""
        return f'{self.__class__.__name__}(width={self.width}, height="{self.height}", ratio={self.ratio}) '
~~~~

* <b>Extra lines augmentation.</b> A huge boost was given by an inserting single or several thin black lines into an image. We reckon that this method helped our model to recognize samples with crossed text better.

![Extra lines augmentation with a single line](https://miro.medium.com/max/469/1*9nwhBkV9rCmGX8kcC9F-Zw.png)

~~~~python
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random

class ExtraLinesAugmentation:
  '''
  Add random black lines to an image
  Args:
    number_of_lines (int): number of black lines to add
    width_of_lines (int): width of lines
  '''

  def __init__(self, number_of_lines: int = 1, width_of_lines: int = 10):
    self.number_of_lines = number_of_lines
    self.width_of_lines = width_of_lines
  
  def __call__(self, img):
    '''
    Args:
      img (PIL Image): image to draw lines on
    Returns:
      PIL Image: image with drawn lines
    '''

    draw = ImageDraw.Draw(img)
    for _ in range(self.number_of_lines):
      x1 = random.randint(0, np.array(img).shape[1]); y1 = random.randint(0, np.array(img).shape[0])
      x2 = random.randint(0, np.array(img).shape[1]); y2 = random.randint(0, np.array(img).shape[0])
      draw.line((x1, y1, x2 + 100, y2), fill=0, width=self.width_of_lines)

    return img
  
  def __repr__(self):
    return f'{self.__class__.__name__}(number_of_lines={self.number_of_lines}, width_of_lines="{self.width_of_lines}")'
~~~~

* <b>Postprocessing.</b> We faced significant issues with predictions of spaces. There were many times out model predicted two parts of one word separately and two words merged. Because of that we used postprocessing with huge dictionaries. We acquired them from parsing all training labels (around 9k words) and then we parsed words from official language models provided by competition hosts (around 160k words as the final result). This tweak works like that:
<br><b>Predicted: we gave you a pre sent
<br>After PP: we gave you a present</b>

* <b>Rotating vertical samples.</b> This method is related only to our case since we had plenty of vertical samples in the training set (with vertically written text as well). So, 90 degree rotation of these images boosted score a lot.

### Model overview

We used seq2seq model (extracting features using CNN as a backbone + encoder-decoder transfromer for generating output). 
In fact, we started with the amazing baseline created by [Vladislav Kramarenko](https://github.com/vlomme/OCR-transformer).
For backbone we chose two main models:
1. ResNeXt101_32x8d
2. DenseNet161

Also, we used AdamW as the optimizer, ReduceLROnPlateau as the scheduler and the simple CrossEntropyLoss. Of course, we tuned some hyperparameters such as learning rate, batch size, dropout and etc.

![Simple scheme of the approach](https://miro.medium.com/max/875/1*bcS1b2viGj9_SKh2HLmeKQ.png)

Finally, we implemented ensemble technique for 3 backbones. We didn’t use our best model with smart resize for this ensemble as 
its submission failed due to the time limit (moreover we kept our 9k dictionary due to the same problem).
1. DenseNet161 (with latest clean samples and smart resize, CER 5.025, Val CER 4.553)
2. ResNeXt101 (with latest clean samples and smart resize, CER 5.047, Val CER 4.750)
3. ResNeXt101(with standard samples and default resize, CER 5.286, Val CER 4.711)

— — — — — — — — — — — — — — — — — — — — — — 
<br>
<b>Final public score: CER 4.789, WER 24.068, String Accuracy 45.157</b>

### Conclusion

In this contest we learned a lot of stuff related to OCR problem. And we hope ideas mentioned in this article will help you to boost your model perfomance.
This concludes our post about AI Journey 2020 Competition.


