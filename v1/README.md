# CliffDelineaTool v1.0 (Original)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5724975.svg)](https://doi.org/10.5281/zenodo.5724975)

**Latest versions**: MATLAB v1.2.0; Python v1.2.0

This folder contains the original CliffDelineaTool algorithm by Zuzanna M. Swirad.

## Citation

If you use this code, please cite:

> Swirad Z.M. & Young A.P. 2022. CliffDelineaTool v1.2.0: an algorithm for identifying coastal cliff base and top positions. Geoscientific Model Development 15: 1499–1512. https://doi.org/10.5194/gmd-15-1499-2022

## About

*CliffDelineaTool* is an algorithm for mapping coastal cliffs by finding cliff base and top positions along cross-shore transects. Written in MATLAB and available in Python, it takes as input text files with series of points containing information on point ID, transect ID, elevation and distance from the seaward transect ends.

Points can contain information on XY coordinates that will be retained for easier incorporation of cliff base and top positions into GIS software. See the `datasets` folder for the calibration and validation datasets.

## How to generate cross-shore transects and points in ArcMap

1. Create polylines to delimit seaward and landward extent of transects
2. Generate equally-spaced points along the seaward polyline (*Generate Points along Lines*)
3. Add a new field to the Attribute Table of the point shapefile (*Calculate Field*: ID = FID + 1)
4. Copy the point shapefile
5. Get the locations of the nearest points along the landward polyline (*Near*; tick 'location')
6. Extract those nearest points (*Make XY Event Layer* of the *Near* location in point Attribute Table)
7. Append the new point layer to the copied point shapefile (*Data Management > Append*; 'no test')
8. Add a new field (*Calculate Field*: ID_1 = ID)
9. Convert points to a polyline (*Points to Line*; field: ID_1)
10. Densify polyline to desired interval (*Densify*) and create point shapefile (*Feature Vertices to Points*)
11. Extract elevation values from DEM (*Extract Values to Points*)
12. Calculate distance to seaward polyline (*Near*)
13. Export the Attribute Table

For repetitive surveys, generate points only once, then update only the elevation.

## How to import outputs into ArcMap

Starting from v1.2.0 it is possible to include XY coordinates as point properties and directly import them as XY layer in GIS software.

For data with no XY information, join the point shapefile with the text file output of *CliffDelineaTool* (*Add Join*; use point ID as the 'join field'). Select points that have any values in the model output columns of the Attribute Table (*Select by Attribute*; e.g. 'Field1>0').

## Authors

- **Zuzanna M. Swirad** (zswirad@ucsd.edu) - Scripps Institution of Oceanography, UC San Diego
- Help in debugging: George Thomas
