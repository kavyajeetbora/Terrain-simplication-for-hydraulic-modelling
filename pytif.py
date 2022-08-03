import rasterio  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import osmnx as ox
import cv2

class Trim_Raster():
    
    def __init__(self, src_file, row_i=None, col_i=None):
        self.src_file = src_file
        self.src_array, self.src = self.read_raster()
        self.row_i = row_i
        self.col_i = col_i
    
    def read_raster(self, summary=True):
        with rasterio.open(self.src_file) as src:
            img_array = src.read(1)
            l,b,r,t = src.bounds
            x_res, y_res = src.res
            source = src

        if summary:
            print("Image size:",img_array.shape)
            print("Bounds:",l,b,r,t)
            print("Resolution:",x_res,y_res)
            print(src.crs)

        return img_array, src

    def clean_raster(self):
        x0,y0 = np.where(np.abs(self.src_array)>9999)
        img_arr = self.src_array.copy()
        img_arr[x0,y0] = 0
        return img_arr

    def trim_data(self, img_arr):
        lat1,lat2 = self.row_i
        long1, long2 = self.col_i
        trim_img = img_arr[lat1:lat2, long1:long2]    
        return trim_img

    def write_raster(self, file_path, img_array):

        l,b,r,t = self.src.bounds
    
        new_dataset = rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=img_array.shape[0],
            width=img_array.shape[1],
            count=1,
            dtype=img_array.dtype,
            crs = self.src.crs,
            transform = rasterio.Affine( ## Transform the pixel coordinates to UTM coordinates
                    0.4, 0, l + self.col_i[0] * self.src.res[0], 0, -0.4, t - self.row_i[0] * self.src.res[1]
                )  
        )
        new_dataset.write(img_array, 1)
        new_dataset.close()   

    def display_image(self, img_array):
        fig,ax = plt.subplots(figsize=(5,5))
        ax.imshow(img_array, cmap = 'gray')
        ax.set_title('Image') 
        plt.xticks([]), plt.yticks([])
        plt.close()
        return fig


def get_street_geometry(north, south, east, west, tags={"highway":True}):
    '''Returns a geodataframe containing the streets geometry within an area extends provided'''
    gdf = ox.geometries.geometries_from_bbox(north, south, east, west, tags=tags)
    gdf = gdf.loc["way"].iloc[:,0:2]
    
    return gdf

def edge_detection(image_array, decimal=2):
    blur = cv2.GaussianBlur(np.abs(image_array)*10**(decimal), ksize=(5,5), sigmaX=0)
    img_mask = cv2.Canny(blur.astype(np.uint8), threshold1=255/3, threshold2=255)
    return img_mask

def extract_edges(img_array, src):
        
    img_mask = edge_detection(img_array)
    x1,y1 = np.where(img_mask>0)
    df = np.column_stack((x1,y1,img_array[x1,y1]))
    df = pd.DataFrame(df, columns=["x","y","z"])
    df[["x","y"]] = df[["x","y"]].astype(np.uint16)

    x_res, y_res = src.res
    l,b,r,t = src.bounds
    x = np.arange(l,r,x_res)
    y = np.arange(t,b,-y_res)

    df["long"] = x[df["y"].values]
    df["lat"] = y[df["x"].values]

    # df = df[["long", "lat", "z"]]
    return df



