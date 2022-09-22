# Terrain simplify 

## Things to do

Since the boundary elements are selected with all the significant slope

Right now we have the out in x,y,z file format. We need to convert it to raster tiff image for it to upload in the terrain analysis tool

1. Delaunay Triangulation 

Questions

1. What is the purpose of compressing 
- To reduce the size of the raster image so as to analyse it
2. What is the procedure for doing the compression ?
    - First select only the pixels that has significant slopes and ignore the other flat points. We will have one x,y,z file
    - Convert the x,y,z file to triangulated irregular network (TIN)
    - Then convert the TIN to raster image through interpolation
