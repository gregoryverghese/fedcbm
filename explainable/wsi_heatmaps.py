import sys
# Updated: No need for sys.path.append with proper package structure
import os
import glob
import math
import argparse
import traceback
from pathlib import Path

import cv2
import openslide
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage import gaussian_filter
from minotaur.wsi.utilities import TissueDetect
from geojson_heatmaps import GeoJson


class Overlay(TissueDetect):

    def __init__(self, thumb, attentions, keys):
        super().__init__(thumb)
        self.wsi_img = thumb
        print(f'thumbnail shape: {thumb.shape}')         
        self.attentions = attentions
        self.info = keys
        self._generate_tissue_contour()
        self.thumb_mask = self.contour_mask
        self._overlay = None
        self.canvas = np.zeros(thumb.shape[:2]) 
        self._heatmap = None
        #self.hm_cmap = self._heatmap()  # self.heatmap_turbo


    @property
    def heatmap(self):
        #heatmap_img = self.hm_cmap()
        self._heatmap = self._get_heatmap()
        return self._heatmap
    

    def _get_heatmap(self):
        canvas = (self.canvas*255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(canvas, cv2.COLORMAP_MAGMA)
        #heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        return heatmap_img


    def _norm_att(self):
        max_a = max(self.attentions)
        min_a = min(self.attentions)
        f = lambda x: (x - min_a)/(max_a - min_a)
        self.attentions = list(map(f, self.attentions))


    def _percentile_clip(self, lower=1, upper=99, eps=1e-8):
        """
        Clips and normalizes values in x to the [0,1] range
        based on the given percentile bounds.

        Parameters
        ----------
        x : array-like
            Input array of attention scores (e.g., raw logits).
        lower : float
            Lower percentile for clipping (default 1).
        upper : float
            Upper percentile for clipping (default 99).
        eps : float
            Small value to avoid division by zero.

        Returns
        -------
        clipped_norm : np.ndarray
            Normalized attention scores in [0,1] after clipping.
        """
        
        low_val = np.percentile(self.attentions, lower)
        high_val = np.percentile(self.attentions, upper)
        self.attentions = np.clip(self.attentions, low_val, high_val)
        self.attentions = (self.attentions - low_val) / (high_val - low_val + eps)

    def _norm_scores(self, x):
        max_a = max(self.attentions)
        min_a = min(self.attentions)
        return (x - min_a)/(max_a - min_a)


    def plot_dist(self):
        self._norm_att()
        data = pd.DataFrame({'scores': self.attentions})
        return data.scores.plot.density(color='green')


    def get_xy(self, key):
        # Example key: "b'101648_58224'"
        return int(key.split('_')[0][2:]), int(key.split('_')[1][:-1])  


    def get_overlay(self, dim, downsample):
        dim = int(dim / math.sqrt(downsample))
        #self._norm_att()
        self._percentile_clip()
        h, w = self.canvas.shape

        for i, a in enumerate(self.attentions):
            tile_heatmap = np.ones((dim, dim)) * a
            x, y = self.get_xy(self.info[i])
            x = int(x / downsample)
            y = int(y / downsample)
            
            x_end = min(x + dim, w)
            y_end = min(y + dim, h)
            tile_w = x_end - x
            tile_h = y_end - y

            if tile_w > 0 and tile_h > 0:
                self.canvas[y:y_end, x:x_end] = tile_heatmap[:tile_h, :tile_w]

        self.canvas = self.canvas
        return self.canvas


    def get_gaussian_heatmap(self, sigma=30, alpha=0.6):
        """
        Generate a Gaussian-smoothed heatmap overlay.

        Parameters:
        - sigma: controls the spread of the Gaussian blur.
        - alpha: controls transparency of the heatmap overlay.

        Returns:
        - overlayed_image: RGB image with heatmap blended over thumbnail.
        """
        attention_values = self.canvas
        print('sigma alpha', sigma, alpha) 
        print('unique values',np.unique(attention_values))
        print('data type', attention_values.dtype)
        smoothed_attention = gaussian_filter(attention_values, sigma=sigma, mode='constant')
        color_map = plt.cm.get_cmap('magma') 
        colors = color_map(smoothed_attention)

        base_img = self.wsi_img
        if base_img.shape[2] == 4:
            base_img = base_img[:, :, :3]
        base_img = base_img.astype(np.uint8)

        overlayed_image = cv2.addWeighted(((colors[:,:,0:3]*255)).astype(np.uint8), 0.5, base_img, 0.5, 0)
        overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
        return overlayed_image


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-wp', '--wsi_path', required = True,
                        help = 'path to a directory of WSIs') 
    parser.add_argument('-as', '--attention_scores_path', required = True,
                        help = 'path to directory containing attention scores as numpy files and IDs as text files')
    parser.add_argument('-sp', '--save_path', required = True,
                        help = 'directory for saving down output')
    parser.add_argument('-td','--tile_dims', default = 224, type = int,
                        help = 'dimensions of tiles w.r.t max resolution')
    parser.add_argument('-ds','--downsampling_level', required = True, type = int,
                        help = 'Magnification level to downsample to. e.g., 0, 1, 2.') 
    parser.add_argument('-si','--sigma', default = 10, type = int,
                        help = 'sigma value for gaussian blur of gaussian heatmap')
    parser.add_argument('-aa','--alpha', default = 0.6, type = float,
                        help = 'alpha value for transparency of gaussian heatmap')    
    args = parser.parse_args()

    wsis = glob.glob(os.path.join(args.wsi_path, '*'))
    #attention_files = glob.glob(os.path.join(args.attention_scores_path, '*.npy'))
    root = Path(args.attention_scores_path)
    attention_files = [str(p) for p in root.rglob('*.npy')]
    print(f'attention_files: {attention_files}')
    ids_files = [str(p) for p in root.rglob('*.txt')]

    for f_path in attention_files:
        name = os.path.basename(f_path).split('_')[0]
        print('\nname: ', name)
        wsi_save_path = os.path.join(os.path.dirname(f_path), 'overlays', name)
        os.makedirs(wsi_save_path,exist_ok=True)
        
        try:
            wsi_path = [p for p in wsis if name in p][0]
            print('wsi_path: ', wsi_path)
            wsi = openslide.OpenSlide(wsi_path)
            id_path = [p for p in ids_files if name in p][0]
            with open(id_path, 'r') as f:
                keys = [line.strip() for line in f.readlines()]
        except Exception as e:
            traceback.print_exc()

        print(f'keys: {keys[0]}')
        
        if not os.path.exists(f_path):
            print(f"Attention scores file not found: {attention_scores_file}")
            continue

        try:
            print(f'f_path: {f_path}')
            attentions = np.load(f_path, allow_pickle=True)
            thumb = wsi.get_thumbnail(wsi.level_dimensions[args.downsampling_level])
            thumb = np.array(thumb.convert('RGB'))
            cv2.imwrite(os.path.join(wsi_save_path, name + '_thumb.png'), thumb)
            print(f'sample keys: {keys[:3]}')

            overlay = Overlay(thumb, attentions, keys)
            scores_dist = overlay.plot_dist()
            plt.savefig(os.path.join(wsi_save_path, name + '_scoresdist.png'))
            downsample = int(wsi.level_downsamples[args.downsampling_level])
            print(f'dowsample factor is: {downsample}')
        
            _ = overlay.get_overlay(args.tile_dims, downsample)
            heatmap = overlay.heatmap
            overlay_heatmap = overlay.get_gaussian_heatmap(args.sigma, args.alpha)
            cv2.imwrite(os.path.join(wsi_save_path, name + '_heatmap.png'), heatmap)
            cv2.imwrite(os.path.join(wsi_save_path, name + '_gaussian.png'), overlay_heatmap)
        except Exception as e:
            traceback.print_exc()
            continue
        
        geopath = os.path.join(wsi_save_path, name + '_heatmap.geojson')
        xys = [overlay.get_xy(key) for key in keys]
        geojson = GeoJson(xys, args.tile_dims, attentions)
        _ = geojson.create_geojson(*geojson.create_polygons(), path=geopath)
