# frvtTestbed
Python code to test NIST frvt accuracy of MUGSHOT &amp; pnas datasets

```
pip install -r requirements.txt
python plot.py
```

* NIST comes with the MUGSHOT dataset for testing:
  * 653 pairs (1306 img ppm files)
* Addition pnas if used for quick verification:
  * 20 pairs (12 Genuine(G) pairs & 8 Imposter(I) pairs)
* Additional models need to be downloaded seperately:
  *geo_vision_5_face_landmarks.dat *dlib LM model*
  *09-02_02-45.pb *tensorflow FR*

## TODO list:
- [x] implement *FD* *LM* *Align* *Crop* *FR* flow for pnas dataset
- [X] extract and save features to pnas.txt
- [ ] add MUGSHOT dataset
- [ ] jupyter notebook (ipynb) to visualize G-I Similarity box scatter chart

