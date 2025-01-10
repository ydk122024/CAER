You can run demo.m and demo_single.m to visualize images in the EMOTIC dataset along with the corresponding annotations.


INSTRUCTIONS for running demo.m and demo_single.m


1.- Download the emotic.zip file that contains the images (3.4GB):

https://drive.google.com/a/uoc.edu/uc?id=0B7sjGeF4f3FYQUVlZ3ZOai1ieEU&export=download

Uncompress the zip file, and keep the folder structure.

2.- Before running demo.m or demo_multiple.m

+ set your own paths on lines 16 and 19, as indicated in the code

+ notice that right after line 10 you can select the split you want to visualize (train, validation, test). 



NOTES:

* Notice that the validation set and the test set contain multiple annotations from different annotators. When running demo_multiple.m you see all the available annotations per image. 

* Notice that demo.m uses the function displayAnnotation_multiple.m and demo_single.m uses displayAnnotation.m

* If you use our data, please cite our paper

Kosti, R., Alvarez, J. M., Recasens, A., Lapedriza, A. "Emotion recognition in context." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017.

@inproceedings{kosti2017emotion,
  title={Emotion recognition in context},
  author={Kosti, Ronak and Alvarez, Jose M and Recasens, Adria and Lapedriza, Agata},
  booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}







