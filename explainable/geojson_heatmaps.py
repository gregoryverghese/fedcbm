"""
Chima Eke 2025
"""
import os
import cv2
import json
import uuid
import numpy as np
import matplotlib.cm as cm

class GeoJson:

    CMAPS = {
        'viridis': 'score_to_viridis',
        'turbo': 'score_to_turbo'
    }

    def __init__(self, xys, dim, weights, cmap=None):
        """
        xy: A list of tuples, each item indicating x and y values w.r.t highest resolution on the WSI
        dim: integer value indicating tile size w.r.t highest resolution on the WSI
        weights: attentions normalised to range [0, 1]
        cmap: string indicating color map to use e.g., viridis, turbo
        """ 
        self.xys = xys
        self.dim = int(dim)
        self.weights = self.norm_weights(weights)
        self.color_map = getattr(self, self.CMAPS[cmap]) if cmap is not None else getattr(self, self.CMAPS['turbo'])

        self.polygons = None
        self.names = None
        self.colors = None


    def norm_weights(self, weights):
        max_a = max(weights)
        min_a = min(weights)
        return [(x - min_a) / (max_a - min_a) for x in weights]


    def score_to_turbo(self, a):
        """
        a: Scalar float value in range [0, 1]
        """
        val = np.uint8([[int(a * 255)]])
        bgr = cv2.applyColorMap(val, cv2.COLORMAP_TURBO)[0][0]   # shape (3,)
        rgb = bgr[::-1]
        return rgb.tolist()


    def score_to_viridis(self, a):
        """
        a: Scalar float value in range [0, 1]
        """
        rgba = cm.viridis(a)
        rgb = [int(255 * c) for c in rgba[:3]]
        # bgr = rgb[::-1]
        return rgb


    def score_to_rgba(self, a):
        """
        a: Scalar float value in range [0, 1]
        
        Return: rgba color.
        From my observation, setting alpha to 1 ensures that polygon outline and fill color are uniform when viewed on QuPath. 
        Otherwise, outline will show a deeper shade of the color than the fill.
        However, without the outline, the fill colors do not stand out enough to easily distinguish hotspots on QuPath.
        """
        rgb_lst  = self.color_map(a)
        rgb_lst.append(0.5)
        return rgb_lst


    def create_polygon(self, xy):
        """
        xy: A tuple or list-like containing x and y
        """
        x, y = int(xy[0]), int(xy[1])

        polygon = [
            [x, y],
            [x + self.dim, y],
            [x + self.dim, y + self.dim],
            [x, y + self.dim],
            [x, y]
        ]

        return polygon


    def create_polygons(self):
        self.polygons =  (self.create_polygon(xy) for xy in self.xys)
        self.names =  (self.create_name(xy) for xy in self.xys)
        self.colors = (self.score_to_rgba(a) for a in self.weights) 
        return self.polygons, self.names, self.colors, self.weights


    def create_name(self, xy):
        x, y = int(xy[0]), int(xy[1])
        return f"tile{x}_{y}"


    def create_geojson(self, polygons, names, colors, scores, path=None):
        
        features = []

        for poly, name, color, score in zip(polygons, names, colors, scores):
            feature = {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [poly]
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": name,
                        "color": color
                    },
                    "measurements": {
                        "Score": score
                    }
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(geojson, f, indent=2)

        return geojson