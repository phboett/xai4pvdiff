# Revealing drivers of green technology adoption through explainable AI

This repository provides the code and the data supporting the paper "Revealing drivers of green technology adoption through explainable AI" by Dorothea Kistinger, Maurizio Titz, Philipp C. Böttcher, Michael T. Schaub, Sandra Venghaus, and Dirk Witthaut.

## Installation 

This project was developed using conda environments via [miniforge](https://github.com/conda-forge/miniforge). The dependencies are collected in [conda_env.yml](./conda_env.yml) and can be installed in the local folder `pyenv` using

```bash
    conda env create --file conda_env.yml -p ./pyenv
```

## Data

### Raw Data

All raw data is openly accessible. The following table gives an overview over all data sources. The supplementary information to our paper gives detailed information on the preprocessing steps we employed to the raw data. 

| Feature | Source | Access date | License | Link | 
| ---- | ------ | ---- |-------------|---------|
|Photovoltaics | Marktstammdatenregister by Bundesnetzagentur ('Gesamtdatenauszug vom Vortag') | 16.11.2022| [Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0)|https://www.marktstammdatenregister.de/MaStR/Datendownload|
|Data on the distribution of different types of vehicles|German Federal Motor Vehicles and Transport Authority| 1.8.2024|[Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0) | https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ27/fz27_202404.xlsx?__blob=publicationFile&v=5 |
|Gridded solar radiation|Joint research Centre of the European Commission|16.12.2022|[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.en)|https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-data-download/sarah-solar-radiation_en|
|Geographical middlepoints of municipalities (used to determine solar radiation observed in municipalities)|Federal Statistical Office and the Statistical Regional Offices|16.12.2022|Reproduction and distribution, even in part, is permitted with citation of the source.|https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugJ/31122019_Auszug_GV.html|
|Charging stations| Bundesnetzagentur | 1.5.2023 | [CC BY-ND 3.0 DE](https://creativecommons.org/licenses/by-nd/3.0/de/deed.de)  | https://www.bundesnetzagentur.de/DE/Fachthemen/ElektrizitaetundGas/E-Mobilitaet/Ladesaeulenkarte |
|Ownership ratio|Census 2011 by Federal Statistical Office and the Statistical Regional Offices|14.12.2022|[Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0)|https://www.zensus2011.de/SharedDocs/Downloads/DE/Pressemitteilung/DemografischeGrunddaten/csv\_GebaudeWohnungen.zip?\_\_blob=publicationFile\&v=2|
|Household sizes|Census 2011 by Federal Statistical Office and the Statistical Regional Offices|14.12.2022|[Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0)|https://www.zensus2011.de/SharedDocs/Downloads/DE/Pressemitteilung/DemografischeGrunddaten/csv\_HaushalteFamilien.zip?\_\_blob=publicationFile\&v=2|
|Various descriptive features|INKAR database by the German Federal Institute for Researchon Building, Urban Affairs and Spatial Development|02.09.2022|[Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0)|https://www.bbr-server.de/imagemap/inkar/download/inkar_2021.zip|
|Results of election for the Bundestag in 2017|Regional Database by Federal Statistical Office and the Statistical Regional Offices|09.11.2022|[Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0)|https://www.regionalstatistik.de/genesis//online?operation=table&code=14111-01-03-5&bypass=true&levelindex=0&levelid=1685362760357#abreadcrumb|
|Changes of AGS keys of municipalities|Federal Statistical Office|04.01.2023|Reproduction and distribution, even in part, is permitted with citation of the source.|https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Namens-Grenz-Aenderung/namens-grenz-aenderung.html|
|Geographic data to display maps of municipalities|Federal Agency for Cartogravphy and Geodesy|27.01.2023|[Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0)|https://daten.gdz.bkg.bund.de/produkte/vg/vg250_ebenen_1231/2020/vg250_12-31.gk3.shape.ebenen.zip|
|Geographic data to display maps of municipal aasociations|Federal Agency for Cartogravphy and Geodesy| 01.06.2023 |[Datenlizenz Deutschland - Namensnennung - Version 2.0](https://www.govdata.de/dl-de/by-2-0)|https://gdz.bkg.bund.de/index.php/default/digitale-geodaten/verwaltungsgebiete/verwaltungsgebiete-1-250-000-stand-31-12-vg250-31-12.html|
|

### Preprocessing Data

The scripts and jupyter notebooks to preprocess the raw data can be found in submodule [preprocessing](./xai_green_tech_adoption/preprocessing/). The supplementary information to our paper gives detailed information on the preprocessing steps we employed. The preprocessed data is placed in the folder *input*. It serves as the input for the subsequent XAI and LASSO analysis. 
Note, that the photovolatic preprocessing is run first, since this prepares most of the socio-economic data. The notebooks contained in the folder [preprocessing/photovoltaics](./xai_green_tech_adoption/preprocessing/photovoltaics) should be executed in the specified order: 

- *handling_changes_in_AGS.ipynb*
- *preprocess_inkar.ipynb*
- *preprocess_census.ipynb*
- *preprocess_regional_database.ipynb*
- *preprocess_radiation_data.ipynb*
- *read_pv_data_xml.ipynb*
- *preprocess_pv_data.ipynb*
- *prepare_input_data.ipynb*

After these scripts were run, the input file for the XAI analysis of the PV adoption (`input.csv`) is placed in `data/input/`. It contains both features and target for the subsequent analysis.
Following, this the data for the analysis of BEV is appended. The methods needed for this analysis can be found in [preprocessing/electric_vehicles](./xai_green_tech_adoption/preprocessing/electric_vehicles/). 
The script `merge_all_preprocessed.py` in this folder contains the meta function to perform the preprocessing for BEVs. These scripts create the file `bev_input.csv` in the aforementioned folder.
Due to the renaming, fusion and splitting up of municipal association during different years, these changes had to be identified, and the resulting knowledge was used in the preprocessing.  

The preprocessed input data and raw data, as well as the output data, can be found online under this [link](https://fz-juelich.sciebo.de/s/PbhS1ucZslnF8Je). 
The folders in this zip-archive should be placed in `data/`.


## XAI Analysis

The pipeline implementing the recursive feature elimination for the GBT models for the BEVs and PV can be found in [rfe.py](./xai_green_tech_adoption/rfe/rfe.py). 
For each GBT simulation, we save the results of the simulation comprising the performances and features included in all GBT models (*data/output/results_rfe\*.csv*).
We provide metadata on simulations in additional files (*data/output/metadata_rfe\*.csv*). 
These include, e.g., the intervals chosen for the hyperparameter optimization. 

For a baseline analysis, we performed simulation runs for LASSO models focussing on different values of the tuning parameter alpha. 
The results of the simulation are given in *benchmarking_lasso.csv* and *benchmarking_lasso_large_alpha.csv*.
The LASSO simulation implemented as a baseline analysis is given in [benchmarking_lasso.py](./xai_green_tech_adoption/rfe/benchmarking_lasso.py).

In addition to the preprocessed input data, the output of the GBT and LASSO simulations can be found online under this [link](https://fz-juelich.sciebo.de/s/PbhS1ucZslnF8Je). The *input* and *output* folder should be placed into the *data* folder.

### Analyses and Visualizing the Results

The plots presented in the article that is currently under review, can be found in [plots_paper.py](./xai_green_tech_adoption/plots_paper.py). 
The last function is a meta function that produces all plots if required output data of the XAI pipeline was produced before. 
The Jupyter notebook [analysis_lasso_model.ipynb](./xai_green_tech_adoption/shap_analysis/analysis_lasso_model.ipynb) implements analyses of the LASSO simulations as a baseline comparison. 
All analyses and visualizations are based on the output of the GBT and LASSO simulations given in the *data/output/* folder. 

In order to plot the maps, we need additional data. 
The folder *maps* gives geographic data of municipalities (the data source is provided in table above) and the folder *intermediate_data* contains a mapping *mapping_municipalities_2000_2019.csv* used to map municipalities identified by their AGS key to the respective municipal associations identified by their ARS key. 
The mapping takes into account all changes in AGS between 2000 and 2019 (data source: see table).

## Contributors

- Dorothea Kistinger ([Orcid](https://orcid.org/0009-0005-7127-1494))
- Philipp C. Böttcher ([Orcid](https://orcid.org/0000-0002-3240-0442))


