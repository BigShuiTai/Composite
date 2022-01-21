import os, cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

__version__ = '0.2'

'''
Composite Module by BigShuitai version 0.2
It supports Pesudo-visible (PVIS) Composite, Pesudo-color (PCOLOR) Composite, and Added-background PVIS Composite
'''

Image.MAX_IMAGE_PIXELS = 2300000000

class Composite:
    def __init__(self, figsize=(10,10), dpi=200):
        self.method_imread = dict(rgb=cv2.COLOR_BGR2RGB, bgr=cv2.IMREAD_COLOR, bgr2=cv2.COLOR_RGB2BGR, rgba=cv2.COLOR_BGRA2RGBA, bgra=cv2.COLOR_RGBA2BGRA)
        self.method_split = dict(rgb='rgb', bgr='bgr', rgba='rgba', bgra='bgra')
        self.interpolation = dict(s=cv2.INTER_AREA, l=cv2.INTER_CUBIC)
        self.central_longitude = 0
        self.mapsize = (21600, 10800)
        self.figsize = figsize
        self.bkg_file = 'nasa.png'
        self.dpi = dpi
    
    @staticmethod
    def _calc_figsize(self, georange):
        DEFAULT_WIDTH = 10
        latmin, latmax, lonmin, lonmax = georange
        ratio = (latmax - latmin) / (lonmax - lonmin)
        figsize = (DEFAULT_WIDTH, DEFAULT_WIDTH * ratio)
        return figsize
    
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
    
    def set_mapsize(self, size):
        self.mapsize = (21600, 10800) if size is None else size
    
    def set_bkg_file(self, file):
        self.bkg_file = 'nasa.png' if file is None else file
    
    def set_central_longitude(self, longitude):
        self.central_longitude = 0 if longitude is None else longitude
    
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
    
    def pvis_composite(self, lons, lats, C7, C13, C15):
        '''
        @parameter lons: longitude data
        @parameter lats: latitude data
        @parameter C7: channel 7 (or similar) data
        @parameter C13: channel 13 (or similar) data
        @parameter C15: channel 15 (or similar) data
        return type: numpy.ndarray of X, Y, Z data
        '''
        sst = 30 * np.cos(lats * np.pi / 180)
        te = (C7[:, 0:-1] - sst[:, 0:-1] + 4.5) / (C13[:, 1:] - sst[:, 1:]) * 0.8
        te2 = (C13[:, 0:-1] - C15[:, 1:]) * 1.25
        te2 = (15 - te2) / 15.5
        syN = -1 * C13[:, 0:-1] / 50 - 2 / 5
        syN[syN < 0] = 0
        syN[syN > 1] = 1
        te = te * (1 - syN) + te2 * syN
        te[te<0] = 0
        data = te
        return lons, lats, data
    
    def pcolor_composite(self, pvisImage=None, infraredImage=None, resize=1, filename=None):
        '''
        @parameter pvisImage: PVIS image for compositing
        @parameter infraredImage: IR image for compositing
        @parameter resize: resize image
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
        elif isinstance(pvisImage, str) and isinstance(infraredImage, str):
            if os.path.isfile(pvisImage) and os.path.isfile(infraredImage):
                # pvis
                ima = Image.open(pvisImage)
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
                img1 = self.PIL_to_CV(ima, 'bgr2')
                
                # ir
                ima = Image.open(infraredImage)
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
                img2 = self.PIL_to_CV(ima, 'bgr2')
            else:
                return False
        else:
            return False
        
        ''' Composite images '''
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
    
    def bg_composite(self, georange, frontImageFile=None, datas=None, shapeline=False, resize=1, filename=None):
        '''
        @parameter datas: X, Y, Z data of tuple for front image for compositing
        @parameter georange: range for cropping background image, if set 'auto', ```datas``` should not None
        @parameter shapeline: if true, it will plot coastlines when plotting data; it always be used for debugging
        @parameter frontImageFile: if ```datas``` is None, it will be used for front image for compositing
        @parameter resize: resize image
        @parameter filename: if it is not none, it will be the target file name for saving image
        return type: numpy.ndarray if ```filename``` is none, or boolen
        '''
        ''' Process front image '''
        if not datas is None:
            lons, lats, data = datas
            if georange == 'auto':
                latmin, latmax, lonmin, lonmax = lats.min(), lats.max(), lons.min(), lons.max()
                self.figsize = self._calc_figsize((latmin, latmax, lonmin, lonmax))
            else:
                latmin, latmax, lonmin, lonmax = georange
            f, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=180)))
            ax.set_extent([lonmin, lonmax, latmin, latmax], crs=ccrs.PlateCarree())
            cmap, vmin, vmax = "gray", 0, 1
            pcolor_kw = dict(cmap=cmap, vmin=vmin, vmax=vmax)
            pcm = plt.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), shading='auto', **pcolor_kw)
            if shapeline:
                import cartopy.feature as cfeature
                ax.add_feature(cfeature.COASTLINE.with_scale("10m"), facecolor="None", edgecolor="white", lw=0.5)
                ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="None", edgecolor="white", lw=0.25)
                ax.add_feature(cfeature.RIVERS.with_scale("10m"), facecolor="None", edgecolor="white", linestyle='-', lw=0.25)
            plt.axis('off')
            f.canvas.draw()
            plt.clf()
            # Get the RGBA buffer from the figure
            w, h = f.canvas.get_width_height()
            buf = np.fromstring(f.canvas.tostring_rgb(), dtype=np.uint8)
            buf.shape = (w, h, 3)
            ima = Image.frombytes("RGB", (w, h), buf.tostring())
            img2 = self.PIL_to_CV(ima, 'bgr2')
        else:
            ima = Image.open(frontImageFile)
            # transform image
            if georange == 'auto':
                return False
            latmin, latmax, lonmin, lonmax = georange
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
        
        ''' Process background image '''
        # Crop background image - from nasdaq's WikiPlot
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
        img1 = self.PIL_to_CV(ima, 'bgr2')
        
        ''' Composite images '''
        # resize image
        if resize > 0:
            rows, cols = img2.shape[:-1]
            img2 = cv2.resize(img2, (int(rows*resize), int(cols*resize)), interpolation=self.interpolation['l'])
            if img1.shape > img2.shape:
                img1 = cv2.resize(img1, img2.shape[:-1][::-1], interpolation=self.interpolation['s'])
            elif img1.shape < img2.shape:
                img1 = cv2.resize(img1, img2.shape[:-1][::-1], interpolation=self.interpolation['l'])
        # get RGB
        color_A, color_B = self.cv_split(img2, 'bgr'), self.cv_split(img1, 'bgr')
        r, g, b = color_A['r'], color_A['g'], color_A['b']
        r1, g1, b1 = color_B['r'], color_B['g'], color_B['b']
        # 柔光
        # rgb转化为0~1的值
        b = b/255+10/255
        b1 = b1/255/2
        g1 = g1/255/2
        r1 = r1/255/2
        # 初步合成
        blue = b*b1*2+np.power((b),1.6)*(1-2*b1)
        green = b*g1*2+np.power((b),1.6)*(1-2*g1)
        red = b*r1*2+np.power((b),1.6)*(1-2*r1)
        # calculate blue marble's total RGB
        brtb = g1+r1
        # 按照底图亮度确定饱和度调整系数，使其针对具有较高底图亮度的陆地
        k = (1+brtb*14)
        # 计算合成后图像的亮度
        brt = (blue+green+red)/3
        # 增加rgb值与亮度的差，增加饱和度
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
