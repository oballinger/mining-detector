# Amazon Gold Mining Detection and Map

Code for the automated detection of (largely illegal) artisanal gold mining in Sentinel-2 satellite imagery; a web map of gold mines in the Amazon rainforest; journalism tied to this work.

<!--![mining-header](https://user-images.githubusercontent.com/13071901/146877405-3ec46c73-cc80-4b1a-8ad1-aeb189bb0b38.jpg)-->
[![mining-header-planet](https://user-images.githubusercontent.com/13071901/146877590-b083eace-2084-4945-b739-0f8dda79eaa9.jpg)](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html)

* [**LAUNCH WEB MAP**](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html)
* [**INTERPRETING THE MAP**](https://github.com/earthrise-media/mining-detector#interpreting-the-map)
* [**JOURNALISM**](https://github.com/earthrise-media/mining-detector#journalism)
* [**METHODOLOGY**](https://github.com/earthrise-media/mining-detector#methodology)

---

## Interpreting the map

The mining of concern here is practiced in every country intersecting the Amazon basin. It is called _artisanal_ because it is typically practiced by individuals or small groups of individuals without heavy machinery, although miners will sometimes source financial backing to bring in an excavator. Miners slash the rainforest to bare earth and then pump water through underlying sediments to liberate the precious metal. They introduce mercury to form an amalgam with the gold, which can then be separated from other particles, and the mercury burnt off due to its relatively low boiling point. Typically miners work along streams and rivers, because of the high demand for water, and because rivers provide access deep into the rainforest.

The environmental and human costs are high. Mining transforms healthy rainforest into biological wasteland of bare earth and toxic wastewater pools. Mercury enters adjacent streams and rivers. In the Amazon basin, miners frequently operate within indigenous lands, bringing with them new diseases and the potential for violent conflict. It has been estimated that upwards of a fifth of Yanomami people died during a mining boom in the 1980s. 

The characteristic mine scar is readily identifiable from satellite. On the banks of a river, you will observe jumbled, multi-colored wastewater pools. They can be brown, tan, different shades of green, and often lurid yellows or turquoise. For the most part they are irregular in size, shape, and orientation. Often nearby you can observe miners' encampments, often some blue-tarped tents, and in well-developed mines a dirt airstrip is cut to fly in miners and fly out the gold. Here are some examples of mines: 

(Five image sequence)

Terrain features that can masquerade as mines include sandy bars in rivers, braided rivers, some unusual badlands, farm ponds for livestock, and especially aquaculture ponds. 

(image sequence)

You can recongnize aquaculture ponds by their geometric shape, efficient use of space, and presence in obvious agricultural zones. There are number of such ponds mistakenly identified as mines on the map, in Bolivia, especially.

The automated detector is a work in progress. We rushed to expand its scope to the whole of the Amazon basin, without sampling all the new terrain features encountered outside our original working domain. While there are some false positive detections on the map, it is not hard for a person to learn to distinguish these from actual mine sites. 

On the whole, though, we are continually surprised by how reliable the detections are despite how widespread. Given some modest human discretion, this should be a useful resource to those interested in tracking mining activity in the region.


### Users should be aware of the following limitations:

**Old Basemap Imagery**

(note that map has two options for satellite imagery)

Mining in the Amazon is growing rapidly. Most basemap imagery in the Amazon is not current, thus some regions classified as containing mines will not appear to have mining activity in the imagery. See example below. Regions in question can be assessed by viewing recent [Sentinel 2 imagery on SentinelHub EO Browser](https://apps.sentinel-hub.com/eo-browser/?zoom=14&lat=-7.13214&lng=-57.36245&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2Fbd86bcc0-f318-402b-a145-015f85b9427e&datasetId=S2L2A&fromTime=2020-09-16T00%3A00%3A00.000Z&toTime=2020-09-16T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR), or Planetscope data accessible through the [Planet NICFI program](https://www.planet.com/nicfi/).
![mining-imagery-comparison](https://user-images.githubusercontent.com/13071901/146989519-d1e537c4-7d70-438d-b4a5-06b2a41a8482.jpg)

**Model Accuracy**

In order to run across the full breadth of the Amazon basin, the model's sensitivity and precision have been reduced in order to improve generalization. This means that the model outputs have some false positives (mining classification where none is present) and false negatives (mines that are rejected by the model). Common false negative failure modes include older mining sites that may be inactive, and edges of active mining regions. False positives can at times be triggered by natural sedimented pools, man made water bodies, and natural earth-scarring activities such as landslides.

The aggregate assessment of mining status should be trusted, but users should attempt to validate results by eye if precise claims of mined regions are needed. The vast majority of classifications are correct, but we cannot validate each of the detections by hand. Given that mining often happens in clusters, isolated detections of mining should be validated more rigorously for false positives.

Additionally, there are a few regions that we could not assess mining activity by eye in high resolution satellite imagery. We have decided to leave these regions in the output data and maps.
![error types](https://user-images.githubusercontent.com/13071901/147019219-98c518fb-72d1-4e35-bf32-9fe058b5d6eb.jpg)


**Area overestimation**

The goal of this work is mine detection rather than area estimation, and our classification operates on the classification of 440 m x 440 m patches. If the network assesses that mining exists anywhere within the patch, then the full patch is declared a mine. This leads to a systematic overestimation of mined area if it is naively computed from the polygon boundaries. Relative year-to-year change calculations are accurate since the polygon area overestimation is consistent.

Building a segmentation model that operates on detected regions is a viable extension of this work.

## Journalism 

![MiningTitlesCollage](https://user-images.githubusercontent.com/11287904/150589512-5d2f1e1c-b946-4f35-90a0-09efbcecc83a.jpg)

This work grew out of a series of collaborations with journalists seeking to expose illegal gold mining activity and document its impacts on the environment and local indigenous communities. At first, we found identified mines by sight in satellite imagery. Then we crowd-sourced image sleuthing with high school students. Finally it made sense to try to automate the identification of mine sites. The training datasets for the machine learned models followed from those early applications of human intelligence.

Reports using the automated detection outputs: 
* [The pollution of illegal gold mining in the Tapajós River](https://infoamazonia.org/en/storymap/the-pollution-of-illegal-gold-mining-in-the-tapajos-river/). The story is part of the _InfoAmazonia_ series, [Murky Waters](https://infoamazonia.org/en/project/murky-waters/), on pollution in the Amazon River system and links to  sargassum seaweed blooms in the Caribbean.
* Forthcoming work with _ArmandoInfo_, _El Pais_, and the Pulitzer Center's Rainforest Investigation Network.

Related reporting: 
* [Amazon gold rush: The threatened tribe](https://graphics.reuters.com/BRAZIL-INDIGENOUS/MINING/rlgvdllonvo/index.html), _Reuters_, 2019, on illegal mining in protected Yanomami Ingigenous Territory.
* [Illegal mining sparks malaria outbreak in indigenous territories in Brazil](https://infoamazonia.org/en/2020/11/25/mineracao-ilegal-contribui-para-surto-de-malaria-em-terras-indigenas-no-para/), _InfoAmazonia_ and _Mongabay_, 2020.
* [Gana por ouro](https://theintercept.com/2021/09/16/mineradora-novata-ja-explorou-32-vezes-mais-ouro-do-que-o-previsto-em-area-protegida-da-amazonia/),  _The Intercept_, 2021. Report on an industrial gold mine operating without environmental permits. Two weeks after the story appeared the mine was shut down and fined the equivalent of two million US dollars.
* [Garimpo destruidor](https://theintercept.com/2021/12/04/garimpo-ilegal-sai-cinza-para-amazonia/), _The Intercept_, 2021.

## Methodology

### Overview

The mine detector is a light-weight convolutional neural network, which we train to discrimate mines from other terrain in the Amazon basin by feeding it hand-labeled examples of mines and other key features as they appear in Sentinel-2 satellite imagery. The network operates on 440 m x 440 m patches of data extracted from the [Sentinel 2 L1C data product](https://sentinel.esa.int/web/sentinel/missions/sentinel-2). Each pixel in the patch captures the light reflected from Earth's surface in twelve bands of visible and infrared light. We average (median composite) the Sentinel data across a four-month period to reduce the presence of clouds, cloud shadow, and other transitory effects. 

During run time, the network assesses each patch for signs of recent mining activity, and then the region of interest is shifted by 140 m for the network to make a subsequent assessment. This process proceeds across the entire region of interest. The network makes 326 million individual assessments in covering the 6.7 million square kilometers of the Amazon basin. 

The system was developed for use in the Amazon, but it has also been seen to work in other tropical biomes.

### Results
#### Assessement of mining in the Amazon basin in 2020

[Amazon mine map](https://earthrise-media.github.io/mining-detector/amazon-mine-map.html) and [dataset](data/outputs/44px_v2.6/mining_amazon_all_unified_thresh_0.8_v44px_v2.6_2020-01-01_2021-02-01_period_4_method_median.geojson). Analysis via the [44px v2.6 model](models/44px_v2.6_2021-11-09.h5).

#### Tapajós basin mining progression, 2016-2020 

[Tapajós mine map](https://earthrise-media.github.io/mining-detector/tapajos-mining-2016-2020pub.html) and [dataset](data/outputs/28_px_v9/28_px_tapajos_2016-2020_thresh_0.5.geojson). In this case, we analyzed the region yearly from 2016-2020 to monitor the growth of mining in the area, using the earlier [28px v9 model](models/28_px_v9.h5). 

#### Hand-validated dectections of mines in Venezuela's Bolívar and Amazonas states in 2020

[Venezuela mine map](https://earthrise-media.github.io/mining-detector/bolivar-amazonas-2020v9verified.html), [Bolívar dataset](data/outputs/28_px_v9/bolivar_2020_thresh_0.8verified.geojson) and [Amazonas dataset](data/outputs/28_px_v9/amazonas_2020_thresh_0.5verified.geojson). Analysis via the 28px v9 model. 

#### Generalization Test in Ghana's Ashanti Region, 2018 and 2020

[Ghana mine map](https://earthrise-media.github.io/mining-detector/ghana-ashanti-2018-2020-v2.8.html) and [dataset](data/outputs/44px_v2.8/mining_ghana_ashanti_v44px_v2.8_2017-2020.geojson). This was a test of the model's ability to generalize to tropical geographies outside of the Amazon basin, using the 44px v2.8 model. 
 
### Running the Code
This repo contains all code needed to generate data, train models, and deploy a model to predict presence of mining in a region of interest. While we welcome external development and use of the code, subject to terms of our open [MIT license](WIP link), creating datasets and deploying the model currently requires access to the [Descartes Labs](https://descarteslabs.com/) platform. 

#### Setup

Download and install [Miniforge Conda env](https://github.com/conda-forge/miniforge/) if not already installed:


| OS      | Architecture          | Download  |
| --------|-----------------------|-----------|
| Linux   | x86_64 (amd64)        | [Miniforge3-Linux-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh) |
| Linux   | aarch64 (arm64)       | [Miniforge3-Linux-aarch64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh) |
| Linux   | ppc64le (POWER8/9)    | [Miniforge3-Linux-ppc64le](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-ppc64le.sh) |
| OS X    | x86_64                | [Miniforge3-MacOSX-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh) |
| OS X    | arm64 (Apple Silicon) | [Miniforge3-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) |

Then run 
```
chmod +x ~/Downloads/Miniforge3-{platform}-{architecture}.sh
sh ~/Downloads/Miniforge3-{platform}-{architecture}.sh
source ~/miniforge3/bin/activate
```

Next, create a conda environment named `mining-detector` by running `conda env create -f environment.yml` from the repo root directory. Activate the environment by running `conda activate mining-detector`. Code has been developed and tested on a Mac with python version 3.9.7. Other platforms and python releases may work, but have not yet been tested.

The data used for model training may be accessed and downloaded from `s3://mining-data.earthrise.media`.

#### Notebooks
The system runs from three core notebooks. 

##### `create_dataset.ipynb` (requires Descartes Labs access)
Given a GeoJSON file of sampling locations, generate a dataset of Sentinel 2 images. Dataset is stored as a pickled list of numpy arrays.

##### `train_model.ipynb`
Train a neural network based on the images stored in the `data/training_data/` directory. Data used to train this model is stored at `s3://mining-data.earthrise.media`.

##### `deploy_model.ipynb` (requires Descartes Labs access)
Given a model file and a GeoJSON describing a region of interest, run the model and download the results. Options exist to deploy the model on a directory of ROI files.

#### Data
The data directory contains two directories.
- `data/boundaries` contains GeoJSON polygon boundaries for regions of interest where the model has been deployed.
- `data/sampling_locations` contains GeoJSON datasets that are used as sampling locations to generate training datasets. Datasets in this directory should be considered "confirmed," and positive/negative class should be indicated in the file's title.

#### Models
The models directory contains keras neural network models saved as `.h5` files. The model names indicate the patch size evaluated by the model, followed by the model's version number and date of creation. Each model file is paired with a corresponding config `.txt` file that logs the datasets used to train the model, some hyperparameters, and the model's performance on the test dataset.

The model `44px_v2.8_2021-11-11.h5` is currently the top performer overall, though some specificity has been sacrificed for generalization. Different models have different strengths/weaknesses. There are also versions of model v2.6 that operate on [RGB](44px_v2.6_rgb_2021-11-11.h5) and [RGB+IR](models/44px_v2.6_rgb_ir_2021-11-11.h5) data. These may be of interest when evaluating whether multispectral data from Sentinel is required.
