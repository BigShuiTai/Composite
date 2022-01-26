import os, cv2
import numpy as np
from scipy import interpolate
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

__version__ = '0.3.0'

'''
Composite Module by BigShuiTai version 0.3.0
It supports Pesudo-visible (PVIS) Composite, Pesudo-color (PCOLOR) Composite, and Added-background PVIS Composite
'''

Image.MAX_IMAGE_PIXELS = 2300000000

class Composite(object):
    def __init__(self, figsize=(10,10), dpi=200):
        '''
        Init variables:
            - method_imread: imread image methods
            - method_split: split RGB data methods
            - interpolation: interpolate methods when resize image
            - mapsize: ```bg_composite``` of background image's size
            - bkg_file: ```bg_composite``` of background image's route & filename
            - dpi: Dots Per Inch, saved image's resolution
        '''
        self.method_imread = dict(rgb=cv2.COLOR_BGR2RGB, bgr=cv2.IMREAD_COLOR, bgr2=cv2.COLOR_RGB2BGR, rgba=cv2.COLOR_BGRA2RGBA, bgra=cv2.COLOR_RGBA2BGRA)
        self.method_split = dict(rgb='rgb', bgr='bgr', rgba='rgba', bgra='bgra')
        self.interpolation = dict(s=cv2.INTER_AREA, l=cv2.INTER_CUBIC)
        self.central_longitude = 0
        self.mapsize = (21600, 10800)
        self.figsize = figsize
        self.bkg_file = 'nasa.png'
        self.dpi = dpi
    
    def set_mapsize(self, size):
        self.mapsize = (21600, 10800) if size is None else size
    
    def set_bkg_file(self, file):
        self.bkg_file = 'nasa.png' if file is None else file
    
    def set_central_longitude(self, longitude):
        self.central_longitude = 0 if longitude is None else longitude
    
    def _calc_figsize(self, georange):
        DEFAULT_WIDTH = 10
        latmin, latmax, lonmin, lonmax = georange
        ratio = (latmax - latmin) / (lonmax - lonmin)
        figsize = (DEFAULT_WIDTH, DEFAULT_WIDTH * ratio)
        return figsize
    
    def _interpolate(self, data, nx=2, ny=2, kx=3, ky=3, autoMask=True):
        '''
        @parameter data: 2-D array of data
        @parameter nx, ny: scale number
        @parameter kx, ky: degrees of the bivariate spline
        @parameter autoMask: whether mask array which has numpy.nan & numpy.inf
        return type: numpy.ndarray
        '''
        if autoMask:
            data = np.ma.masked_where(np.isnan(data), data)
            data = np.ma.masked_where(np.isinf(data), data)
        out_size_x = 1 / nx
        out_size_y = 1 / ny
        nx, ny = data.shape
        x = np.arange(nx)
        y = np.arange(ny)
        z = data
        interp = interpolate.RectBivariateSpline(x, y, z)
        xx = np.arange(0, nx, out_size_x)
        yy = np.arange(0, ny, out_size_y)
        ndata = interp(xx, yy)
        return ndata
    
    def convert(self, latlon, intro):
        '''
        @parameter latlon: latitude or longitude, int / float type
        @parameter intro: string for calculating position of pixel
        return type: int
        '''
        if intro == 'lon':
            if latlon >= self.central_longitude:
                pixel = int((latlon - self.central_longitude) / 360.0 * self.mapsize[0])
            else:
                pixel = int((latlon + self.central_longitude) / 360.0 * self.mapsize[0])
        elif intro == 'lat':
            pixel = int((90.0 - latlon) / 180.0 * self.mapsize[1])
        return pixel
    
    def imread(self, image, method=None):
        '''
        @parameter image: File route
        @parameter method: 'rgb' or 'bgr'
        return type: cv2 class
        '''
        method = self.method_imread['rgb'] if method is None else self.method_imread[method]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, method)
        return image
    
    def cv_split(self, cv_img, method=None):
        '''
        @parameter cv_img: cv2 class
        @parameter method: 'rgb' or 'bgr'
        return type: dict
        '''
        method = self.method_split['rgb'] if method is None else self.method_split[method]
        if method == 'rgb':
            r, g, b = cv2.split(cv_img)
            color = dict(r=r, g=g, b=b)
        elif method == 'bgr':
            b, g, r = cv2.split(cv_img)
            color = dict(b=b, g=g, r=r)
        elif method == 'rgba':
            b, g, r, a = cv2.split(cv_img)
            color = dict(b=b, g=g, r=r, a=a)
        elif method == 'bgra':
            b, g, r, a = cv2.split(cv_img)
            color = dict(b=b, g=g, r=r, a=a)
        else:
            color = ()
        return color
    
    def PIL_to_CV(self, PIL_img, method=None):
        '''
        @parameter PIL_img: PIL class
        @parameter method: 'rgb' or 'bgr'
        return type: cv2 class
        '''
        method = self.method_imread['rgb'] if method is None else self.method_imread[method]
        image = cv2.cvtColor(np.asarray(PIL_img), method)
        return image
    
    def CV_to_PIL(self, CV_img, method=None):
        '''
        @parameter CV_img: cv2 class
        @parameter method: 'rgb' or 'bgr'
        return type: PIL class
        '''
        method = self.method_imread['rgb'] if method is None else self.method_imread[method]
        image = Image.fromarray(cv2.cvtColor(CV_img, method))
        return image
    
    def pvis_composite(self, lats, datas, mode='geostationary', HCC=True):
        '''
        @parameter lats: latitude data for calculating linear SST
        @parameter datas: data for pvis-compositing
        @parameter mode: pvis-compositing mode, str of 'geostationary', 'viirs', 'slstr', 'modis' or 'avhrr'
        @parameter HCC: whether using high cloud correction method
        return type: numpy.ndarray
        '''
        mode = mode.lower()
        if mode not in ('geostationary', 'viirs', 'slstr', 'modis', 'avhrr'):
            return np.array([])
        # composite formula by @Carl & @hhui-mt
        from matplotlib.colors import Normalize, LinearSegmentedColormap
        cfdata = {'green': [(0, 0, 0), (1, 1, 1)], 'red': [(0, 0, 0), (1, 1, 1)], 'blue': [(0, 0, 0), (1, 1, 1)]}
        norm = Normalize(vmin=0, vmax=1, clip=True)
        SM = plt.cm.ScalarMappable(norm, LinearSegmentedColormap('1', cfdata))
        sst = 30 * np.cos(lats * np.pi / 180)
        if mode == 'geostationary':
            # it is applied to geostationary satellite's PVIS composite,
            # etc HIMAWARI-8/9, GOES-16/17, METEOSAT-8/9/10/11 and so on
            C7, C13, C15 = datas
            d1, d2, d3 = C7, C13, C15
        elif mode == 'modis':
            # it is applied to the satellite,
            # which has MODIS (Moderate Resolution Imaging Spectroradiometer)'s PVIS composite,
            # etc TERRA, AQUA and so on
            C20, C31, C32 = datas
            d1, d2, d3 = C20, C31, C32
        elif mode == 'slstr':
            # it is applied to the satellite,
            # which has SLSTR (Sea and Land Surface Temperature Radiometer)'s PVIS composite,
            # etc Sentinel-3A/3B and so on
            S7, S8, S9 = datas
            d1, d2, d3 = S7, S8, S9
        elif mode == 'viirs':
            # it is applied to the satellite,
            # which has VIIRS (Visible Infrared Imaging Radiometer Suite)'s PVIS composite,
            # etc JPSS, Suomi NPP and so on
            I4, I5, M15, M16 = datas
            d1, d2, d3, d4 = I4, I5, M15, M16
        elif mode == 'avhrr':
            # it is applied to the satellite,
            # which has AVHRR (Advanced Very High Resolution Radiometer)'s PVIS composite,
            # etc NOAA-6/8/10/15/16/17/18/19, MetOp-A/B and so on
            B3B, B4, B5 = datas
            d1, d2, d3 = B3B, B4, B5
        if mode in ('geostationary', 'modis', 'slstr', 'avhrr'):
            te = (d1[:, 0:-1] - sst[:, 0:-1] + 4.5) / (d2[:, 1:] - sst[:, 1:]) * 0.8
            te2 = (d2[:, 0:-1] - d3[:, 1:]) * 1.25
            te2 = (15 - te2) / 15.5
            syN = -1 * d2[:, 0:-1] / 50 - 2 / 5
            if HCC:
                # high cloud correction - by @Carl
                d23Diff_HCC = d2[:, 1:] - d3[:, 1:]
                d23Diff_HCC = SM.to_rgba(d23Diff_HCC)[:,:,0]
                te = np.power(te, 2-d23Diff_HCC)*np.power((2-d23Diff_HCC),1.35)
        elif mode == 'viirs':
            d34Diff = d3[:, 0:-1] - d4[:, 1:]
            te = (d1[:, 1:-1] - sst[:, 1:-1] + 4.5) / (d2[:, 2:] - sst[:, 2:]) * 0.8
            te2 = d34Diff * 1.25
            te2 = (15 - te2) / 15.5
            syN = -1 * d2[:, 1:-1] / 50 - 2 / 5
            if HCC:
                # high cloud correction - by @Carl
                d34Diff_HCC = d3[:, 1:] - d4[:, 1:]
                d34Diff_HCC = SM.to_rgba(d34Diff_HCC)[:,:,0]
                d34Diff_HCC = self._interpolate(d34Diff_HCC, nx=2, ny=2)
                te = np.power(te, 2-d34Diff_HCC)*np.power((2-d34Diff_HCC),1.35)
        syN[syN < 0] = 0
        syN[syN > 1] = 1
        te2 = SM.to_rgba(te2)[:,:,0]
        te = SM.to_rgba(te)[:,:,0]
        syN = SM.to_rgba(syN)[:,:,0]
        if mode == 'viirs':
            # WARNING: M-BAND's resolution is smaller than I-BAND's,
            # so it's necessary to interpolate M-BAND's data before composite
            te2 = self._interpolate(te2, nx=2, ny=2)
        te = te * (1 - syN) + te2 * syN
        data = te
        return data
    
    def pcolor_composite(self, pvisImage=None, infraredImage=None, resize=1, invert=False, filename=None):
        '''
        @parameter pvisImage: PVIS image for compositing
        @parameter infraredImage: IR image for compositing
        @parameter resize: resize image
        @parameter invert: invert pre-load image for compositing correctly
        @parameter filename: if it is not none, it will be the target file name for saving image
        return type: numpy.ndarray if ```filename``` is none, or boolen
        '''
        ''' process images '''
        if isinstance(pvisImage, matplotlib.figure.Figure) and isinstance(infraredImage, matplotlib.figure.Figure):
            # pvis
            f = pvisImage
            f.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = f.canvas.get_width_height()
            buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (w, h, 3)
            ima = Image.frombytes("RGB", (w, h), buf.tostring())
            if invert:
                ima = ImageOps.invert(ima)
            img1 = self.PIL_to_CV(ima, 'bgr2')
            
            # ir
            f = infraredImage
            f.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = f.canvas.get_width_height()
            buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (w, h, 3)
            ima = Image.frombytes("RGB", (w, h), buf.tostring())
            if invert:
                ima = ImageOps.invert(ima)
            img2 = self.PIL_to_CV(ima, 'bgr2')
        elif isinstance(pvisImage, str) and isinstance(infraredImage, str):
            if os.path.isfile(pvisImage) and os.path.isfile(infraredImage):
                # pvis
                ima = Image.open(pvisImage)
                if invert:
                    rgb = ima.split()
                    if len(rgb) == 4:
                        r, g, b, a = rgb
                        ima = Image.merge('RGB', (r, g, b))
                    ima = ImageOps.invert(ima)
                img1 = self.PIL_to_CV(ima, 'bgr2')
                
                # ir
                ima = Image.open(infraredImage)
                if invert:
                    rgb = ima.split()
                    if len(rgb) == 4:
                        r, g, b, a = rgb
                        ima = Image.merge('RGB', (r, g, b))
                    ima = ImageOps.invert(ima)
                img2 = self.PIL_to_CV(ima, 'bgr2')
            else:
                return False
        else:
            return False
        
        ''' Composite images '''
        # img1 - pvis image; img2 - ir image
        # resize image
        if resize > 0:
            rows, cols = img2.shape[:-1]
            img2 = cv2.resize(img2, (int(rows*resize), int(cols*resize)), interpolation=self.interpolation['l'])
            if img1.shape > img2.shape:
                img1 = cv2.resize(img1, img2.shape[:-1][::-1], interpolation=self.interpolation['s'])
            elif img1.shape < img2.shape:
                img1 = cv2.resize(img1, img2.shape[:-1][::-1], interpolation=self.interpolation['l'])
        # get RGB
        color_A, color_B = self.cv_split(img1, 'bgr'), self.cv_split(img2, 'bgr')
        r, g, b = color_A['r'], color_A['g'], color_A['b']
        r1, g1, b1 = color_B['r'], color_B['g'], color_B['b']
        # merge image
        dst = cv2.merge([b1, g, r])
        
        ''' Save figure '''
        if not filename is None:
            if not filename.endswith('.png') and not filename.endswith('.jpg'):
                filename += '.png'
            cv2.imwrite(filename, dst * 255)
            return True
        else:
            return dst
    
    def bg_composite(self, georange, pvisImage=None, infraredImage=None, resize=1, filename=None):
        '''
        @parameter georange: range for cropping background image
        @parameter pvisImage: PVIS image for compositing
        @parameter infraredImage: IR image for compositing
        @parameter resize: resize image
        @parameter filename: if it is not none, it will be the target file name for saving image
        return type: numpy.ndarray if ```filename``` is none, or boolen
        '''
        latmin, latmax, lonmin, lonmax = georange
        ''' Process front image '''
        if isinstance(pvisImage, matplotlib.figure.Figure) and isinstance(infraredImage, matplotlib.figure.Figure):
            # pvis
            f = pvisImage
            f.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = f.canvas.get_width_height()
            buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (w, h, 3)
            ima = Image.frombytes("RGB", (w, h), buf.tostring())
            img1 = self.PIL_to_CV(ima, 'bgr2')
            
            # ir
            f = infraredImage
            f.canvas.draw()
            # Get the RGBA buffer from the figure
            w, h = f.canvas.get_width_height()
            buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (w, h, 3)
            ima = Image.frombytes("RGB", (w, h), buf.tostring())
            img2 = self.PIL_to_CV(ima, 'bgr2')
        elif os.path.isfile(pvisImage) and os.path.isfile(infraredImage):
            # pvis
            ima = Image.open(pvisImage)
            f, ax = plt.subplots(figsize=self.figsize)
            plt.axis('off')
            ax.imshow(ima)
            f.canvas.draw()
            plt.clf()
            # Get the RGBA buffer from the figure
            w, h = f.canvas.get_width_height()
            buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (w, h, 3)
            ima = Image.frombytes("RGB", (w, h), buf.tostring())
            img1 = self.PIL_to_CV(ima, 'bgr2')
                
            # ir
            ima = Image.open(infraredImage)
            f, ax = plt.subplots(figsize=self.figsize)
            plt.axis('off')
            ax.imshow(ima)
            f.canvas.draw()
            plt.clf()
            # Get the RGBA buffer from the figure
            w, h = f.canvas.get_width_height()
            buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (w, h, 3)
            ima = Image.frombytes("RGB", (w, h), buf.tostring())
            img2 = self.PIL_to_CV(ima, 'bgr2')
        else:
            return False
        
        ''' Process background image '''
        # Crop background image - from WikiPlot by @nasdaq
        if lonmax <= 180 or lonmin >= 180:
            ima = Image.open(self.bkg_file)
            box = (self.convert(lonmin,'lon'), self.convert(latmax,'lat'), self.convert(lonmax,'lon'), self.convert(latmin,'lat'))
            ima = ima.crop(box)
        else:
            pixelleft, pixelright = self.convert(lonmin, 'lon'), self.convert(lonmax, 'lon')
            upper, lower = self.convert(latmax, 'lat'), self.convert(latmin, 'lat')
            ima = Image.new('RGB', (self.mapsize[0] - pixelleft + pixelright, lower - upper))
            imafrom = Image.open(self.bkg_file)
            apart = imafrom.crop((pixelleft, upper, self.mapsize[0], lower))
            bpart = imafrom.crop((0, upper, pixelright, lower))
            ima.paste(apart, (0,0))
            ima.paste(bpart, (self.mapsize[0] - pixelleft, 0))
        # transform image
        f, ax = plt.subplots(figsize=self.figsize)
        plt.axis('off')
        ax.imshow(ima)
        f.canvas.draw()
        plt.clf()
        # Get the RGBA buffer from the figure
        w, h = f.canvas.get_width_height()
        buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (w, h, 3)
        ima = Image.frombytes("RGB", (w, h), buf.tostring())
        img3 = self.PIL_to_CV(ima, 'bgr2')
        
        ''' Composite images '''
        # img1 - pvis image; img2 - ir image; img3 - background image
        # resize image
        if resize > 0:
            rows, cols = img1.shape[:-1]
            img1 = cv2.resize(img1, (int(rows*resize), int(cols*resize)), interpolation=self.interpolation['l'])
            img2 = cv2.resize(img2, (int(rows*resize), int(cols*resize)), interpolation=self.interpolation['l'])
            if img3.shape > img1.shape:
                img3 = cv2.resize(img3, img1.shape[:-1][::-1], interpolation=self.interpolation['s'])
            elif img3.shape < img1.shape:
                img3 = cv2.resize(img3, img1.shape[:-1][::-1], interpolation=self.interpolation['l'])
        # get RGB
        color_A, color_B, color_C = self.cv_split(img1, 'bgr'), self.cv_split(img2, 'bgr'), self.cv_split(img3, 'bgr')
        b = color_A['b'] / 255 + 15 / 255
        r1, g1, b1 = color_C['r'] / 255 / 2, color_C['g'] / 255 / 2, color_C['b'] / 255 / 2
        b2 = color_B['r'] / 255
        # composite formula by @Carl
        blue = b*b1*2+np.power((b),1.6)*(1-2*b1)
        green = b*g1*2+np.power((b),1.6)*(1-2*g1)
        red = b*r1*2+np.power((b),1.6)*(1-2*r1)
        brtb = g1+r1
        k = 1+brtb*15*(1-b2)
        brt = (blue+green+red)/3
        b = 255/255*brt+(blue-brt)*k+0/255
        g = 255/255*brt+(green-brt)*k+0/255
        r = brt+(red-brt)*k
        # merge image
        dst = cv2.merge([b, g, r])
        
        ''' Save figure '''
        if not filename is None:
            if not filename.endswith('.png') and not filename.endswith('.jpg'):
                filename += '.png'
            cv2.imwrite(filename, dst * 255)
            return True
        else:
            return dst
