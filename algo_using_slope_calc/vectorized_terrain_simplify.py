import numpy as np
import time
import os
from itertools import product
import rasterio
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

def read_raster(file_path):
    '''
    Reads a raster image file (.tif file) and returns the relevant information including the image array
    '''
    with rasterio.open(file_path) as src:
        img_array = src.read(1)
        l,b,r,t = src.bounds
        x_res, y_res = src.res
    return img_array, l,b,r,t, x_res, y_res

def invalid_points(raster_array):
    '''
    For image not in square or rectangle shape, when converted to array fills up the empty elevation values with -3.4028235e+38
    hence this has to be removed 
    '''
    max_val = 9999
    ## Find the index of the valid coordinates:
    x0,y0 = np.where(np.abs(raster_array)>max_val)

    return x0,y0

def adjacent_values(A, index=(1,1)):
    '''
    Finds the matrix containing the neighboring value shifted by some index
    For Example (0,0) index will return a matrix containg all the top-left values
    '''
    B = np.pad(A,pad_width=1)[index[0]:index[0]+A.shape[0],index[1]:index[1]+A.shape[1]]
    return B

def vectorized_slope_calc(A, invalid_pts):   
    
    '''
    Calculates the slope of a value in a matrix with respect to its neighboring values.
    Currently the kernel size chosen is around 3X3 matrix. 
    
    INPUT: 
    1. Takes a matrix of which slope is to be calculated
    Returns: the slope matrix of the same size as the input matrix
    '''
    if min(A.shape) >= 3:

        print("Preparing offset arrays required for slope calculation...")
        ## List down all the indices of the adjacent values:
        p = product(list(range(0,3)),repeat=2) ## Kernel size taken as 3
        indices = []
        for x in p:
            if x != (1,1):
                indices.append(x)
        
        print("Calculating the slope...", end="")
        ## Calculate the slope:
        slope = np.zeros(A.shape)
        for i in indices:
            B = adjacent_values(A, index=i) ## Find the adjacent matrix
            slope += np.abs(A - B)  ## Only find the magnitude of the difference

        ## Clean the raster image 
        x0,y0 = invalid_pts
        slope[x0,y0] = 0
        max_slope_value = slope[slope<1e38].max()
        slope[slope>1e38] = max_slope_value
        print("Done")

        return slope
    
    else:
        print("Array size too small. It should be greater than or equal to (3,3)")
        return None

def simplify_raster(img_array, S, threshold=0.8, grid_size = 20):

    '''
    Simplify the raster image data by reducing the number of points
    And only keeping the pixel data that are having significant slopes

    PARAMETERS
    -------------
    1. img_array - original raster image array
    2. S - Slope matrix of the original raster image
    XXXX  3. ivc - invalid coordinates XXXX 
    4. threshold - Threshold standard deviation, below which points are removed
    5. grid_size - takes data of pixels spaced as per grid size irrespective of slope values,
                    to ensure not all data is removed

    RETURNS
    -----------
    A xyz csv file containing all the important points of the given raster image
    '''

    print("Standardising the slope values")
    scaler = MinMaxScaler()
    ## Normalize the slope values
    Y = scaler.fit_transform(S)
    print("Done!")

    ## indices of slope values with absolute standard deviation more than 1
    print("Thresholding the slopes...", end="")
    xi,yi = np.where(Y>=threshold)
    data = np.column_stack((xi,yi,img_array[xi,yi]))
    data = pd.DataFrame(data, columns=["x","y","z"])
    data[["x","y"]] = data[["x","y"]].astype(np.uint16)
    print("Selected Significant slopes !")

    ## Generate the sparse grid points
    print("Generating sparse grid points...",end="")
    ai = np.arange(0,img_array.shape[0], grid_size)
    bi = np.arange(0,img_array.shape[1], grid_size)
    Ai, Bi = np.meshgrid(ai,bi)
    depth_values = img_array[Ai,Bi]
    grid = np.column_stack((Ai.ravel(), Bi.ravel(), depth_values.ravel()))
    grid = pd.DataFrame(grid, columns=["x","y","z"])
    grid[["x","y"]] = grid[["x","y"]].astype(np.uint16)
    grid = grid[grid["z"].abs()<1e10].reset_index(drop=True)
        
    ## merge the significant slope points and the grid points
    output = pd.concat([grid, data]).drop_duplicates(subset=["x","y"])

    print("Total number of points taken: %d" % output.shape[0])
    print("Percentage points taken: %.2f %% of the total" % (output.shape[0]/img_array.ravel().shape[0]*100))
    return output

def display_compression(img_array, output):

    '''
    Displays the significant pixels overlaid on the original image. 
    Also displays the amount of compression done in percentage
    '''
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(25,25)

    fraction = output.shape[0]/len(img_array.ravel())*100

    axes[0].imshow(img_array, cmap="gray")
    axes[1].imshow(img_array, cmap="gray")
    axes[1].scatter(output['y'],output["x"], s=0.01, c="red")
    axes[0].axis("off")
    axes[0].set_title("Original tiff file with size %d by %d pixels" % (img_array.shape[0], img_array.shape[1]))
    axes[1].axis("off")
    axes[1].set_title("Pixels with significant slopes (%.1f %% of the total number of pixels i.e. %d points)" % (fraction, output.shape[0]))
    return fig

## main function
if __name__ == "__main__":
    '''
    Pipeline for image pixel simplification using vectorization:
    -----------------------------------------
    '''
    start_time = time.time()

    filename = "Danmark_Hav_Nakskov.tif"
    directory = "Data\\NAKSKOV\\RAW-DATA-INPUT-TIFF"

    # filename = "SierraLeone_hilly.tif"
    # directory = "Data\\10mX10m_HillyArea_SL"

    # filename = "Denmark_urban.tif"
    # directory = "Data\\40cmX40cm_UrbanArea_DK"

    file_path = os.path.join(directory,filename)
    print("Reading the raster image....", end="")
    raster_array, l,b,r,t, x_res, y_res = read_raster(file_path)
    print("Done")

    print("Cleaning the raster image...", end="")
    invalid_pts =  invalid_points(raster_array)
    print("Done")

    print("Input image size:", raster_array.shape)

    print("Calculating the slope values for each pixel...", end="")
    S = vectorized_slope_calc(raster_array, invalid_pts)
    ## Thresholding the slopes -> Only consider slopes to be significant only if it is more than 10 %tile.
    output = simplify_raster(raster_array, S, threshold=0.01, grid_size = 100)  
    print(output.shape)
    print("Done !")
    
    del(raster_array)

    ## convert to coordinates to x,y,z points
    print("Converting the pixel coordinates to UTM coordinates...", end="")
    x = np.arange(l,r,x_res)
    y = np.arange(t,b,-y_res)

    output["long"] = x[output["y"].values]
    output["lat"] = y[output["x"].values]
    print("Done")
    
    print("Exporting the significant slope locations...", end="")
    ## save the xyz data as csv file
    output_directory = "C:\\Users\\kvba\\OneDrive - Ramboll\\Projects\\Terrain smoothing for hydraulic modelling\\Output"
    output[["long", "lat", "z"]].to_csv(os.path.join(output_directory,filename.split(".")[0]+"_threshold=0.05.csv"), index=False)
    print("Done")
    
    end_time = time.time()
    print("Time taken around %.2f" % (end_time-start_time))