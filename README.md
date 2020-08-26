# frvtTestbed
Python code to test NIST frvt accuracy of MUGSHOT &amp; pnas datasets

```
pip install -r requirements.txt
python plot.py
```

* NIST comes with the MUGSHOT dataset for testing:
  * 653 pairs (1306 img ppm files)
* pnas is also used for quick verification:
  * 20 pairs (12 Genuine(G) pairs & 8 Imposter(I) pairs)
* Additional tensorflow FR model need to be downloaded seperately:
  * 09-02_02-45.pb

## TODO list:
- [x] implement *FD* *LM* *Align* *Crop* *FR* flow for pnas dataset
- [X] extract and save features to pnas.txt
- [X] visualize G-I Similarity box scatter chart
- [X] add MUGSHOT dataset
- [X] add pnas dataset
- [X] cleanup code (use dataframe instead of list)

## G-I Similarity box scatter chart
* MUGSHOT:

![Alt text](mugshotGIboxPlot.png?raw=true "Title")

* pnas:

![Alt text](pnasGIboxPlot.png?raw=true "Title")

