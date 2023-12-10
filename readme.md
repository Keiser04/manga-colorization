UPD. See the [improved version](https://github.com/qweasdd/manga-colorization-v2).

# Improvements

1. A saving method was created to save the state of the optimizer along with the model weights. This way, you can also recall the state of the optimization (such as learning rates and momentum) when you load the checkpoint.
2. Until now only one GPU could be used but in theory with train2.py it would be possible to enter with 2 or more.
3. 3 notebooks have been created for training in both Kaggle and Colab (2) and another one for painting a manga.
4. A script "go-color.bat" was created to install everything needed for local coloring.

# In-progress

1. Until now only one GPU could be used but in theory with train2.py it would be possible to enter with 2 or more.
2. notebooks have been created for training in both Kaggle and Colab (2) and another one for painting a manga.
3. A script "go-color.bat" was created to install everything needed for local coloring.

# Automatic colorization

1. Download [generator](https://drive.google.com/file/d/1Oo6ycphJ3sUOpDCDoG29NA5pbhQVCevY/view?usp=sharing),  [extractor](https://drive.google.com/file/d/12cbNyJcCa1zI2EBz6nea3BXl21Fm73Bt/view?usp=sharing) and [denoiser ](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view?usp=sharing) weights. Put generator and extractor weights in `model` and denoiser weights in `denoising/models`.
2. To colorize image, folder of images, `.cbz` or `.cbr` file, use the following command:
```
$ python inference.py -p "path to file or folder"
```

# Manual colorization with color hints

1. Download [colorizer](https://drive.google.com/file/d/1BERrMl9e7cKsk9m2L0q1yO4k7blNhEWC/view?usp=sharing) and [denoiser ](https://drive.google.com/file/d/161oyQcYpdkVdw8gKz_MA8RD-Wtg9XDp3/view?usp=sharing) weights. Put colorizer weights in `model` and denoiser weights in `denoising/models`.
2.  Run gunicorn server with:
```
$ ./run_drawing.sh
```
3. Open `localhost:5000` with a browser.

# References
1. Extractor weights are taken from https://github.com/blandocs/Tag2Pix/releases/download/release/model.pth
2. Denoiser weights are taken from http://www.ipol.im/pub/art/2019/231.
