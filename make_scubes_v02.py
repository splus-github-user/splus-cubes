# -*- coding: utf-8 -*-
"""
 Tool to produce calibrated cubes from S-PLUS images
 Herpich F. R. fabiorafaelh@gmail.com - 2021-03-09

 based/copied/stacked from Kadu's scripts

 ***
 Edited and adapted to python3.* - 2023-03-06
"""
from __future__ import print_function, division

import os
import glob
import itertools
import warnings

import numpy as np
from scipy.interpolate import RectBivariateSpline
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
import astropy.constants as const
from astropy.wcs import FITSFixedWarning
from astropy.wcs.utils import skycoord_to_pixel as sky2pix
from regions import PixCoord, CirclePixelRegion
from tqdm import tqdm
import sewpy
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
import datetime

from photutils import DAOStarFinder

warnings.simplefilter('ignore', category=FITSFixedWarning)

initext = """
    ===================================================================
                    make_scubes
             This version is not yet completely debugged
             In case of crashes, please send the log to:
                Herpich F. R. fabiorafaelh@gmail.com
    ===================================================================
    """


class SCubes(object):
    def __init__(self):
        """basic definitions"""

        self.galaxies = np.array(['NGC1087'])
        self.coords = [['02:46:25.15', '-00:29:55.45']]
        self.tiles = ['STRIPE82-0059']
        self.sizes = np.array([100])
        self.angsize = None
        self.work_dir: str = os.getcwd()
        self.data_dir: str = os.path.join(self.work_dir, 'data/')
        self.zpcorr_dir: str = os.path.join(self.work_dir, 'data/zpcorr_idr3/')
        self.tiles_dir = None
        self.zp_table = 'iDR4_zero-points.csv'
        # self.sexpath = '/home/herpich/bin/sextractor/src/sex'

        # SExtractor contraints
        self.satur_level: float = 1600.0  # use 1600 for elliptical
        self.back_size: int = 54  # use 54 for elliptical or 256 for spiral
        self.detect_thresh = 1.1

        # from Kadu's context
        self.ps = 0.55 * u.arcsec / u.pixel
        self.bands = ['U', 'F378', 'F395', 'F410', 'F430', 'G', 'F515', 'R',
                      'F660', 'I', 'F861', 'Z']
        self.narrow_bands = ['F378', 'F395', 'F410', 'F430', 'F515', 'F660', 'F861']
        self.broad_bands = ['U', 'G', 'R', 'I', 'Z']
        self.bands_names = {'U': "$u$", 'F378': "$J378$", 'F395': "$J395$",
                            'F410': "$J410$", 'F430': "$J430$", 'G': "$g$",
                            'F515': "$J515$", 'R': "$r$", 'F660': "$J660$",
                            'I': "$i$", 'F861': "$J861$", 'Z': "$z$"}
        self.wave_eff = {"F378": 3770.0, "F395": 3940.0, "F410": 4094.0,
                         "F430": 4292.0, "F515": 5133.0, "F660": 6614.0,
                         "F861": 8611.0, "G": 4751.0, "I": 7690.0, "R": 6258.0,
                         "U": 3536.0, "Z": 8831.0}
        self.exptimes = {"F378": 660, "F395": 354, "F410": 177,
                         "F430": 171, "F515": 183, "F660": 870, "F861": 240,
                         "G": 99, "I": 138, "R": 120, "U": 681,
                         "Z": 168}
        self.names_correspondent = {'U': 'u', 'F378': 'J0378', 'F395': 'J0395',
                                    'F410': 'J0410', 'F430': 'J0430', 'G': 'g',
                                    'F515': 'J0515', 'R': 'r', 'F660': 'J0660',
                                    'I': 'i', 'F861': 'J0861', 'Z': 'z'}

    def make_stamps_splus(self, redo=False, img_types=None, bands=None, savestamps=True):
        """  Produces stamps of objects in S-PLUS from a table of names,
        coordinates.

        Parameters
        ----------
        names: np.array
            Array containing the name/id of the objects.

        coords: astropy.coordinates.SkyCoord
            Coordinates of the objects.

        size: np.array
            Size of the stamps (in pixels)

        outdir: str
            Path to the output directory. If not given, stamps are saved in the
            current directory.

        redo: bool
            Option to rewrite stamp in case it already exists.

        img_types: list
            List containing the image types to be used in stamps. Default is [
            "swp', "swpweight"] to save both the images and the weight images
            with uncertainties.

        bands: list
            List of bands for the stamps. Defaults produces stamps for all
            filters in S-PLUS. Options are 'U', 'F378', 'F395', 'F410', 'F430', 'G',
            'F515', 'R', 'F660', 'I', 'F861', and 'Z'.

        savestamps: boolean
            If True, saves the stamps in the directory outdir/object.
            Default is True.


        """
        names = np.atleast_1d(self.galaxies)
        sizes = np.atleast_1d(self.sizes)
        if len(sizes) == 1:
            sizes = np.full(len(names), sizes[0])
        sizes = sizes.astype(np.int)
        img_types = ["swp", "swpweight"] if img_types is None else img_types
        work_dir = os.getcwd() if self.work_dir is None else self.work_dir
        tile_dir = os.getcwd() if self.tiles_dir is None else os.path.join(self.tiles_dir, self.tiles[0])
        header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                       "INSTRUME", "AIRMASS"]
        bands = self.bands if bands is None else bands

        # Selecting tiles from S-PLUS footprint
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
              '- Selecting tile names from the S-PLUS footprint')
        cols = [self.galaxies,
                np.transpose(self.coords)[0],
                np.transpose(self.coords)[1],
                np.array(self.tiles)]
        names = ['NAME', 'RA', 'DEC', 'TILE']
        fields = Table(cols, names=names)

        # Producing stamps
        for field in tqdm(fields, desc="Fields"):
            field_name = field["TILE"]
            fnames = [field['NAME']]
            fcoords = SkyCoord(ra=field['RA'], dec=field['DEC'],
                               unit=(u.hour, u.deg))  # self.coords[idx]
            fsizes = np.array(sizes)[fields['NAME'] == fnames]
            stamps = dict((k, []) for k in img_types)
            for img_type in tqdm(img_types, desc="Data types", leave=False,
                                 position=1):
                for band in tqdm(bands, desc="Bands", leave=False, position=2):
                    #tile_dir = os.path.join(tile_dir, field["TILE"], band)
                    fitsfile = os.path.join(tile_dir, "{}_{}.fits".format(
                        field["TILE"], img_type))
                    try:
                        header = fits.getheader(fitsfile)
                        data = fits.getdata(fitsfile)
                    except:
                        fzfile = os.path.join(tile_dir, field['TILE'] + '_' + band + '_' + img_type + '.fz')
                        f = fits.open(fzfile)[1]
                        header = f.header
                        data = f.data
                    else:
                        failedfile = os.path.join(tile_dir, "{}_{}.fits".format(field["TILE"], img_type))
                        Warning('file %s not found' % failedfile)
                    wcs = WCS(header)
                    xys = wcs.all_world2pix(fcoords.ra, fcoords.dec, 1)
                    for i, (name, size) in enumerate(tqdm(zip(fnames, fsizes),
                                                          desc="Galaxies", leave=False, position=3)):
                        galdir = os.path.join(work_dir, name)
                        output = os.path.join(galdir,
                                              "{0}_{1}_{2}_{3}x{3}_{4}.fits".format(
                                                  name, field_name, band, size, img_type))
                        if os.path.exists(output) and not redo:
                            continue
                        try:
                            cutout = Cutout2D(data, position=fcoords,
                                              size=size * u.pixel, wcs=wcs)
                        except ValueError:
                            continue
                        if np.all(cutout.data == 0):
                            continue
                        hdu = fits.ImageHDU(cutout.data)
                        for key in header_keys:
                            if key in header:
                                hdu.header[key] = header[key]
                        hdu.header["TILE"] = hdu.header["OBJECT"]
                        hdu.header["OBJECT"] = name
                        if img_type == "swp":
                            hdu.header["NCOMBINE"] = (header["NCOMBINE"], "Number of combined images")
                            hdu.header["EFFTIME"] = (header["EFECTIME"], "Effective exposed total time")
                        if "HIERARCH OAJ PRO FWHMMEAN" in header:
                            hdu.header["PSFFWHM"] = header["HIERARCH OAJ PRO FWHMMEAN"]
                        hdu.header["X0TILE"] = (xys[0].item(), "Location in tile")
                        hdu.header["Y0TILE"] = (xys[1].item(), "Location in tile")
                        hdu.header.update(cutout.wcs.to_header())
                        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
                        if savestamps:
                            if not os.path.exists(galdir):
                                os.mkdir(galdir)
                            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                                  'Saving', output)
                            hdulist.writeto(output, overwrite=True)
                        else:
                            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                                  'To be implemented')
                            # stamps[img_type].append(hdulist)

    def make_det_stamp(self, savestamp: bool=True):
        """Cut the detection stamp of the same size as the cube stamps to be used for the mask"""

        names = np.atleast_1d(self.galaxies)
        sizes = np.atleast_1d(self.sizes)
        if len(sizes) == 1:
            sizes = np.full(len(names), sizes[0])
        sizes = sizes.astype(np.int)
        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        tile_dir = os.getcwd() if self.tiles_dir is None else os.path.join(self.tiles_dir, self.tiles[0])
        header_keys = ["OBJECT", "FILTER", "EXPTIME", "GAIN", "TELESCOP",
                       "INSTRUME", "AIRMASS"]

        # Selecting tiles from S-PLUS footprint
        cols = [self.galaxies,
                np.transpose(self.coords)[0],
                np.transpose(self.coords)[1],
                np.array(self.tiles)]
        names = ['NAME', 'RA', 'DEC', 'TILE']
        fields = Table(cols, names=names)

        # Producing stamps
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
              'Creating detection stamp for stars detection')
        for field in fields:
            field_name = field["TILE"]
            fnames = [field['NAME']]
            fcoords = SkyCoord(ra=field['RA'], dec=field['DEC'], unit=(u.hour, u.deg))
            fsizes = np.array(sizes)[fields['NAME'] == fnames]
            for i, (name, size) in enumerate(zip(fnames, fsizes)):
                galdir = os.path.join(outdir, name)
                if not os.path.isdir(galdir):
                    os.makedirs(galdir)
                doutput = os.path.join(galdir,
                                       "{0}_{1}_{2}x{2}_{3}.fits".format(
                                           name, field_name, size, 'detection'))
                if not os.path.isfile(doutput):
                    try:
                        d = fits.open(os.path.join(tile_dir, field_name + '_detection.fits'))[0]
                    except:
                        d = fits.open(os.path.join(tile_dir, field_name + '_detection.fits.fz'))[1]
                    else:
                        Warning('Detection image not found. Using rSDSS...')
                        d = fits.open(os.path.join(self.tiles_dir, self.tiles[0], field_name, field_name + '_R_swp.fz'))
                    dheader = d.header
                    ddata = d.data
                    wcs = WCS(dheader)
                    xys = wcs.all_world2pix(fcoords.ra, fcoords.dec, 1)
                    dcutout = Cutout2D(ddata, position=fcoords,
                                       size=size * u.pixel, wcs=wcs)
                    hdu = fits.ImageHDU(dcutout.data)
                    for key in header_keys:
                        if key in dheader:
                            hdu.header[key] = dheader[key]
                    hdu.header["TILE"] = hdu.header["OBJECT"]
                    hdu.header["OBJECT"] = name
                    if "HIERARCH OAJ PRO FWHMMEAN" in dheader:
                        hdu.header["PSFFWHM"] = dheader["HIERARCH OAJ PRO FWHMMEAN"]
                    hdu.header["X0TILE"] = (xys[0].item(), "Location in tile")
                    hdu.header["Y0TILE"] = (xys[1].item(), "Location in tile")
                    hdu.header.update(dcutout.wcs.to_header())
                    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
                    if savestamp:
                        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                              'Saving', doutput)
                        hdulist.writeto(doutput, overwrite=True)

    def get_zps(self, tile: str=None, zp_dir: str=None, zp_table: str=None):
        """ Load zero points."""
        _dir = self.work_dir if zp_dir is None else zp_dir
        tile = self.tiles[0] if tile is None else tile
        zp_table = self.zp_table if zp_table is None else zp_table

        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
              'Reading ZPs table:', os.path.join(_dir, zp_table))
        zpt = pd.read_csv(os.path.join(_dir, zp_table))
        zpt.columns = [a.replace('ZP_', '') for a in zpt.columns]
        zptab = zpt[zpt['Field'] == tile]

        return zptab

    def get_zp_correction(self):
        """ Get corrections of zero points for location in the field. """
        x0, x1, nbins = 0, 9200, 16
        xgrid = np.linspace(x0, x1, nbins + 1)
        zpcorr = {}
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
              'Getting ZP corrections for the S-PLUS bands...')
        for band in self.bands:
            corrfile = os.path.join(self.zpcorr_dir, 'SPLUS_' + band + '_offsets_grid.npy')
            corr = np.load(corrfile)
            zpcorr[band] = RectBivariateSpline(xgrid, xgrid, corr)

        return zpcorr

    def calibrate_stamps(self, galaxy: str=None):
        """
        Calibrate stamps
        """

        galaxy = self.galaxies[0] if galaxy is None else galaxy
        tile = os.listdir(self.work_dir + galaxy)[0].split('_')[1] if self.tiles is None else self.tiles[0]
        zps = self.get_zps(tile=tile)
        zpcorr = self.get_zp_correction()
        stamps = sorted([_ for _ in os.listdir(self.work_dir + galaxy) if _.endswith("_swp.fits")])
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
              'Calibrating stamps...')
        for stamp in stamps:
            filename = os.path.join(self.work_dir, galaxy, stamp)
            h = fits.getheader(filename, ext=1)
            h['TILE'] = tile
            filtername = h["FILTER"]
            zp = float(zps[self.names_correspondent[filtername]].item())
            x0 = h["X0TILE"]
            y0 = h["Y0TILE"]
            zp += round(zpcorr[filtername](x0, y0)[0][0], 5)
            fits.setval(filename, "MAGZP", value=zp,
                        comment="Magnitude zero point", ext=1)

    def run_sex(self, f: object, galaxy: str=None, tile: str=None, size: int=None):
        """ Run SExtractor to the detection stamp """

        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        galaxy = galaxy if self.galaxies is None else self.galaxies[0]
        tile = tile if self.tiles is None else self.tiles[0]
        size = size if self.sizes is None else self.sizes[0]
        galdir = os.path.join(outdir, galaxy)  # if self.gal_dir is None else self.gal_dir
        pathdetect = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'detection'))
        pathtoseg = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'segmentation'))
        sexoutput = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
            galaxy, tile, size, 'sexcat'))

        gain = f[1].header['GAIN']
        fwhm = f[1].header['PSFFWHM']

        # output params for SExtractor
        params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
                  "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO", "FWHM_IMAGE",
                  "CLASS_STAR"]

        # configuration for SExtractor photometry
        config = {
            "DETECT_TYPE": "CCD",
            "DETECT_MINAREA": 4,
            "DETECT_THRESH": self.detect_thresh,
            "ANALYSIS_THRESH": 3.0,
            "FILTER": "Y",
            "FILTER_NAME": os.path.join(self.data_dir, "sex_data/tophat_3.0_3x3.conv"),
            "DEBLEND_NTHRESH": 64,
            "DEBLEND_MINCONT": 0.0002,
            "CLEAN": "Y",
            "CLEAN_PARAM": 1.0,
            "MASK_TYPE": "CORRECT",
            "PHOT_APERTURES": 5.45454545,
            "PHOT_AUTOPARAMS": '3.0,1.82',
            "PHOT_PETROPARAMS": '2.0,2.73',
            "PHOT_FLUXFRAC": '0.2,0.5,0.7,0.9',
            "SATUR_LEVEL": self.satur_level,
            "MAG_ZEROPOINT": 20,
            "MAG_GAMMA": 4.0,
            "GAIN": gain,
            "PIXEL_SCALE": 0.55,
            "SEEING_FWHM": fwhm,
            "STARNNW_NAME": os.path.join(self.data_dir, 'sex_data/default.nnw'),
            "BACK_SIZE": self.back_size,
            "BACK_FILTERSIZE": 7,
            "BACKPHOTO_TYPE": "LOCAL",
            "BACKPHOTO_THICK": 48,
            "CHECKIMAGE_TYPE": "SEGMENTATION",
            "CHECKIMAGE_NAME": pathtoseg
        }

        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
              'Running SExtractor for config:')
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', config)
        sew = sewpy.SEW(workdir=galdir, config=config, sexpath="sex", params=params)
        sewcat = sew(pathdetect)
        sewcat["table"].write(sexoutput, format="fits", overwrite=True)

        return sewcat

    def run_DAOfinder(self, fdata: object):
        "calculate photometry using DAOfinder"

        mean, median, std = 0, 0, 0.5
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', ('mean', 'median', 'std'))
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', (mean, median, std))
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Running DAOfinder...')
        daofind = DAOStarFinder(fwhm=4.0, sharplo=0.2, sharphi=0.9,
                                roundlo=-0.5, roundhi=0.5, threshold=5. * std)
        sources = daofind(fdata)
        return sources

    def make_Lupton_colorstamp(self, galaxy: str=None, tile: str=None, size: int=None):
        """Make Lupton colour image from stamps"""

        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        galaxy = self.galaxies[0] if galaxy is None else galaxy
        tile = self.tiles[0] if tile is None else tile
        size = self.sizes[0] if size is None else size
        galdir = os.path.join(outdir, galaxy)
        bands = self.bands
        blues = ['G', 'U', 'F378', 'F395', 'F410', 'F430']
        greens = ['R', 'F660', 'F515']
        reds = ['I', 'F861', 'Z']

        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Creating Lupton RGB stamps...')
        bimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
            galaxy, tile, band, size)) for band in blues]
        bdata = sum([fits.getdata(img) for img in bimgs])
        gimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
            galaxy, tile, band, size)) for band in greens]
        gdata = sum([fits.getdata(img) for img in gimgs])
        rimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
            galaxy, tile, band, size)) for band in reds]
        rdata = sum([fits.getdata(img) for img in rimgs])
        gal = os.path.join(galdir, f"{galaxy}_{tile}_{size}x{size}.png")
        rgb = make_lupton_rgb(rdata, gdata, bdata, minimum=-0.01, Q=20, stretch=0.9, filename=gal)
        ax = plt.imshow(rgb, origin='lower')

        return rgb

    def calc_main_circle(self, f=None, centralPixCoords=None, angsize=None, size=None, galaxy=None, tile=None):
        """Calculate main circle to identify the galaxy"""

        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        galaxy = self.galaxies[0] if galaxy is None else galaxy
        tile = self.tiles[0] if tile is None else tile
        galdir = os.path.join(outdir, galaxy)
        # contraints = f[1].data < abs(np.percentile(f[1].data, 2.3))
        fdata = f[1].data.copy()
        wcs = WCS(f[1].header)
        ix, iy = np.meshgrid(np.arange(fdata.shape[0]), np.arange(fdata.shape[1]))
        distance = np.sqrt((ix - centralPixCoords[0])**2 + (iy - centralPixCoords[1])**2)
        expand = True
        interaction = 1
        while expand:
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'Iteration number = ', interaction, '; angsize =', angsize)
            innerMask = distance <= angsize
            diskMask = (distance > angsize) & (distance <= angsize + 5)
            outerMask = distance > angsize + 5
            # fmask[innerMask] = 1
            innerPercs = np.percentile(fdata[innerMask], [16, 50, 84])
            diskPercs = np.percentile(fdata[diskMask], [16, 50, 84])
            outerPercs = np.percentile(fdata[outerMask], [16, 50, 84])
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'Inner: [16, 50, 84]', innerPercs)
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'Disk: [16, 50, 84]', diskPercs)
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'Outer: [16, 50, 84]', outerPercs)
            plt.ioff()
            ax1 = plt.subplot(111, projection=wcs)
            # make colour image
            ax1.imshow(fdata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
            circregion = CirclePixelRegion(center=PixCoord(centralPixCoords[0], centralPixCoords[1]),
                                           radius=angsize)
            circregion.plot(color='y', lw=1.5, ax=ax1, label='%.1f pix' % angsize)
            outcircregion = CirclePixelRegion(center=PixCoord(centralPixCoords[0], centralPixCoords[1]),
                                           radius=angsize + 5)
            outcircregion.plot(color='g', lw=1.5, ax=ax1, label='%.1f pix' % (angsize + 5))
            ax1.set_title('RGB')
            ax1.set_xlabel('RA')
            ax1.set_ylabel('Dec')
            ax1.legend(loc='upper left')

            if diskPercs[1] <= outerPercs[1] + (outerPercs[1] - outerPercs[0]):
                path2fig: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.png".format(galaxy, tile, size, 'defCircle'))
                print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                      'Saving fig after finishing iteration:', path2fig)
                plt.savefig(path2fig, format='png', dpi=180)
                plt.close()
                expand = False
            else:
                angsize += 5
                print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                      'Current angsize is:', angsize, '; size/2 is:', size / 2)
                if angsize >= size / 2:
                    plt.show()
                    raise ValueError('Iteration stopped. Angsize %.2f bigger than size %i' % (angsize, size / 2))
                interaction += 1

        fmask = np.zeros(f[1].data.shape)
        fmask[distance <= angsize] = 1
        return circregion, fdata * fmask

    def calc_masks(self, galaxy: str=None, tile: str=None, size: int=None, savemask: bool=False, savefig: bool=False,
                   maskstars: list=[], angsize: float=None, runDAOfinder: bool=False):
        """
        Calculate masks for S-PLUS stamps. Masks will use the catalogue of stars and from the
        SExtractor segmentation image. Segmentation regions need to be manually selected
        """

        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Starting masks...')
        galcoords = SkyCoord(ra=self.coords[0][0], dec=self.coords[0][1], unit=(u.hour, u.deg))
        outdir = os.getcwd() if self.work_dir is None else self.work_dir
        galaxy = self.galaxies[0] if galaxy is None else galaxy
        tile = self.tiles[0] if tile is None else tile
        size = self.sizes[0] if size is None else size
        angsize = self.angsize / 0.55 if angsize is None else angsize / 0.55
        galdir = os.path.join(outdir, galaxy)
        pathdetect: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(galaxy, tile, size, 'detection'))

        # get data
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
              'Getting detection stamp for photometry...')
        f = fits.open(pathdetect)
        ddata = f[1].data
        wcs = WCS(f[1].header)
        centralPixCoords = sky2pix(galcoords, wcs)

        fdata = ddata.copy()
        constraints = ddata > abs(np.percentile(ddata, 2.3))
        fmask = np.zeros(ddata.shape)
        fmask[constraints] = 1
        fdata *= fmask

        # calculate big circle
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Getting galaxy circle...')
        circregion, maskeddata = self.calc_main_circle(f=f, centralPixCoords=centralPixCoords, angsize=angsize,
                                                       size=size, galaxy=galaxy, tile=tile)

        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
              'Running SExtractor to get photometry...')
        sewcat = self.run_sex(f, galaxy=galaxy, tile=tile, size=size)
        sewpos = np.transpose((sewcat['table']['X_IMAGE'], sewcat['table']['Y_IMAGE']))
        radius = 3.0 * (sewcat['table']['FWHM_IMAGE'] / 0.55)
        sidelim = 80
        mask = sewcat['table']['CLASS_STAR'] > 0.1
        mask &= (sewcat['table']['X_IMAGE'] > sidelim)
        mask &= (sewcat['table']['X_IMAGE'] < fdata.shape[0] - sidelim)
        mask &= (sewcat['table']['Y_IMAGE'] > sidelim)
        mask &= (sewcat['table']['Y_IMAGE'] < fdata.shape[1] - sidelim)
        mask &= (sewcat['table']['FWHM_IMAGE'] > 0)
        sewregions = [CirclePixelRegion(center=PixCoord(x, y), radius=z)
                      for (x, y), z in zip(sewpos[mask], radius[mask])]

        # DAOfinder will be needed only in extreme cases of crowded fields
        if runDAOfinder:
            daocat = self.run_DAOfinder(fdata)
            daopos = np.transpose((daocat['xcentroid'], daocat['ycentroid']))
            daorad = 4 * (abs(daocat['sharpness']) + abs(daocat['roundness1']) + abs(daocat['roundness2']))
            daoregions = [CirclePixelRegion(center=PixCoord(x, y), radius=z)
                          for (x, y), z in zip(daopos, daorad)]

        #plt.figure(figsize=(10, 10))
        plt.rcParams['figure.figsize'] = (12.0, 10.0)
        plt.ion()

        # draw subplot 1
        ax1 = plt.subplot(221, projection=wcs)
        # make colour image
        rgb = self.make_Lupton_colorstamp(galaxy=galaxy, tile=tile, size=size)
        # draw coloured image
        ax1.imshow(rgb, origin='lower')
        # draw large circle around the galaxy
        circregion.plot(color='y', lw=1.5)
        # draw small circles around sources selected from SExtractor catalogue
        for sregion in sewregions:
            sregion.plot(ax=ax1, color='g')
        # DAOfinder will be needed only in extreme cases of crowded fields
        if runDAOfinder:
            for dregion in daoregions:
                dregion.plot(ax=ax1, color='m')
        ax1.set_title('RGB')
        ax1.set_xlabel('RA')
        ax1.set_ylabel('Dec')

        # draw subplot 2
        ax2 = plt.subplot(222, projection=wcs)
        # draw gray scaled image
        ax2.imshow(fdata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
        # draw large circle around the galaxy
        circregion.plot(color='y', lw=1.5)
        # draw small circles around sources selected from SExtractor catalogue
        for n, sregion in enumerate(sewregions):
            sregion.plot(ax=ax2, color='g')
            ax2.annotate(repr(n), (sregion.center.x, sregion.center.y), color='green')
        # DAOfinder will be needed only in extreme cases of crowded fields
        if runDAOfinder:
            for dregion in daoregions:
                dregion.plot(ax=ax2, color='m')
        ax2.set_title('Detection')
        ax2.set_xlabel('RA')
        ax2.set_ylabel('Dec')

        # draw subplot 3
        ax3 = plt.subplot(223, projection=wcs)
        # draw gray scaled masked image
        ax3.imshow(maskeddata, cmap='Greys_r', origin='lower', vmin=-0.1, vmax=3.5)
        # draw large circle around the galaxy
        circregion.plot(color='y', lw=1.5)
        # draw small circles around sources selected from SExtractor catalogue
        # mask sources using the size of the FWHM obtained by SExtractor
        for n, sregion in enumerate(sewregions):
            sregion.plot(ax=ax3, color='g')
            if n not in maskstars:
                mask = sregion.to_mask()
                if (min(mask.bbox.extent) < 0) or (max(mask.bbox.extent) > size):
                    print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                          'Region is out of range for extent', mask.bbox.extent)
                else:
                    _slices = (slice(mask.bbox.iymin, mask.bbox.iymax), slice(mask.bbox.ixmin, mask.bbox.ixmax))
                    print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                          mask.bbox.extent, 'min:', min(mask.bbox.extent), _slices)
                    maskeddata[_slices] *= 1 - mask.data
        # DAOfinder will be needed only in extreme cases of crowded fields
        if runDAOfinder:
            for dregion in daoregions:
                dregion.plot(ax=ax2, color='m')
        ax3.set_title('Masked')
        ax3.set_xlabel('RA')
        ax3.set_ylabel('Dec')

        ax4 = plt.subplot(224, projection=wcs)
        # create the mask
        fitsmask = f.copy()
        fitsmask[1].data = np.zeros(maskeddata.shape)
        fitsmask[1].data[maskeddata > 0] = 1
        # draw gray scaled mask
        ax4.imshow(fitsmask[1].data, cmap='Greys_r', origin='lower')
        ax4.set_title('Mask')
        ax4.set_xlabel('RA')
        ax4.set_ylabel('Dec')

        plt.subplots_adjust(wspace=.05, hspace=.2)

        # prepare mask to save and added to the cube
        fitsmask[1].header['IMGTYPE'] = ("MASK", "boolean mask")
        del fitsmask[1].header['EXPTIME']
        del fitsmask[1].header['FILTER']
        del fitsmask[1].header['GAIN']
        del fitsmask[1].header['PSFFWHM']
        if savemask:
            path2mask: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
                galaxy, tile, size, 'mask'))
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Saving mask to', path2mask)
            fitsmask.writeto(path2mask, overwrite=True)

        if savefig:
            path2fig: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.png".format(
                galaxy, tile, size, 'maskMosaic'))
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Saving fig to', path2fig)
            plt.savefig(path2fig, format='png', dpi=180)

        return fitsmask

    def make_cubes(self, galdir: str=None, redo: bool=False, dodet: bool=True, get_mask: bool=True,
                   bands=None, specz="", photz="", maskstars: float=None, bscale: float=1e-19):
        """ Get results from cutouts and join them into a cube. """

        galcoords = SkyCoord(ra=self.coords[0][0], dec=self.coords[0][1], unit=(u.hour, u.deg))
        galaxy = "{}_{}".format(galcoords.ra.value, galcoords.dec.value) if self.galaxies[0] is None else self.galaxies[0]
        galdir = os.path.join(self.work_dir, galaxy) if galdir is None else galdir
        tile = self.tiles[0]
        size = self.sizes[0]

        if not os.path.isdir(galdir):
            os.mkdir(galdir)
        filenames = glob.glob(galdir + '/{0}_{1}_*_{2}x{2}_swp*.fits'.format(galaxy, tile, size))
        while len(filenames) < 24:
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'Calling make_stamps_splus()...')
            self.make_stamps_splus(redo=True)
            filenames = glob.glob(galdir + '/{0}_{1}_*_{2}x{2}_swp*.fits'.format(galaxy, tile, size))
            if len(filenames) < 24:
                raise IOError('Tile file missing for stamps')

        if redo:
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'Calling make_stamps_splus()...')
            self.make_stamps_splus(redo=True)
            filenames = glob.glob(galdir + '/*_swp*.fits')
        if dodet:
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'Calling make_det_stamp()...')
            self.make_det_stamp()
        elif get_mask and not dodet:
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                  'For mask detection image is required. Overwriting dodet=True')
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Calling make_det_stamp()...')
            self.make_det_stamp()

        fields = set([_.split("_")[-4] for _ in filenames]) if self.tiles is None else self.tiles
        sizes = set([_.split("_")[-2] for _ in filenames]) if self.sizes is None else self.sizes
        bands = self.bands if bands is None else bands
        wave = np.array([self.wave_eff[band] for band in bands]) * u.Angstrom
        flam_unit = u.erg / u.cm / u.cm / u.s / u.AA
        fnu_unit = u.erg / u.s / u.cm / u.cm / u.Hz
        imtype = {"swp": "DATA", "swpweight": "WEIGHTS"}
        hfields = ["GAIN", "PSFFWHM", "DATE-OBS"]
        print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Starting cube assembly...')
        for tile, size in itertools.product(fields, sizes):
            cubename = os.path.join(galdir, "{0}_{1}_{2}x{2}_cube.fits".format(galaxy, tile,
                                                                               size))
            if os.path.exists(cubename) and not redo:
                print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Cube exists!')
                continue
            # Loading and checking images
            imgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swp.fits".format(
                galaxy, tile, band, size)) for band in bands]
            # Checking if images have calibration available
            headers = [fits.getheader(img, ext=1) for img in imgs]
            if not all(["MAGZP" in h for h in headers]):
                print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                      'Starting stamps calibration...')
                print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                      'Calling calibrate_stamps()...')
                self.calibrate_stamps()
            headers = [fits.getheader(img, ext=1) for img in imgs]
            # Checking if weight images are available
            wimgs = [os.path.join(galdir, "{0}_{1}_{2}_{3}x{3}_swpweight.fits".format(
                galaxy, tile, band, size)) for band in bands]
            has_errs = all([os.path.exists(_) for _ in wimgs])
            # Making new header with WCS
            h = headers[0].copy()
            w = WCS(h)
            nw = WCS(naxis=3)
            nw.wcs.cdelt[:2] = w.wcs.cdelt
            nw.wcs.crval[:2] = w.wcs.crval
            nw.wcs.crpix[:2] = w.wcs.crpix
            nw.wcs.ctype[0] = w.wcs.ctype[0]
            nw.wcs.ctype[1] = w.wcs.ctype[1]
            try:
                nw.wcs.pc[:2, :2] = w.wcs.pc
            except:
                pass
            h.update(nw.to_header())
            # Performin calibration
            m0 = np.array([h["MAGZP"] for h in headers])
            gain = np.array([h["GAIN"] for h in headers])
            effexptimes = np.array([h["EFFTIME"] for h in headers])
            del h["FILTER"]
            del h["MAGZP"]
            del h["NCOMBINE"]
            del h["EFFTIME"]
            del h["GAIN"]
            del h["PSFFWHM"]
            f0 = np.power(10, -0.4 * (48.6 + m0))
            data = np.array([fits.getdata(img, 1) for img in imgs])
            fnu = data * f0[:, None, None] * fnu_unit
            flam = fnu * const.c / wave[:, None, None] ** 2
            flam = flam.to(flam_unit).value / bscale
            if has_errs:
                weights = np.array([fits.getdata(img, 1) for img in wimgs])
                dataerr = 1.0 / weights + np.clip(data, 0, np.infty) / gain[:, None, None]
                fnuerr = dataerr * f0[:, None, None] * fnu_unit
                flamerr = fnuerr * const.c / wave[:, None, None] ** 2
                flamerr = flamerr.to(flam_unit).value / bscale
            # Making table with metadata
            tab = []
            tab.append(bands)
            tab.append([self.wave_eff[band] for band in bands])
            tab.append(effexptimes)
            names = ["FILTER", "WAVE_EFF", "EXPTIME"]
            for f in hfields:
                if not all([f in h for h in headers]):
                    continue
                tab.append([h[f] for h in headers])
                names.append(f)
            tab = Table(tab, names=names)
            # Producing data cubes HDUs.
            hdus = [fits.PrimaryHDU()]
            hdu1 = fits.ImageHDU(flam, h)
            hdu1.header["EXTNAME"] = ("DATA", "Name of the extension")
            hdu1.header["SPECZ"] = (specz, "Spectroscopic redshift")
            hdu1.header["PHOTZ"] = (photz, "Photometric redshift")
            hdus.append(hdu1)
            if has_errs:
                hdu2 = fits.ImageHDU(flamerr, h)
                hdu2.header["EXTNAME"] = ("ERRORS", "Name of the extension")
                hdus.append(hdu2)
            for hdu in hdus:
                hdu.header["BSCALE"] = (bscale, "Linear factor in scaling equation")
                hdu.header["BZERO"] = (0, "Zero point in scaling equation")
                hdu.header["BUNIT"] = ("{}".format(flam_unit),
                                       "Physical units of the array values")
            if get_mask:
                path2mask: str = os.path.join(galdir, "{0}_{1}_{2}x{2}_{3}.fits".format(
                    galaxy, tile, size, 'mask'))
                if os.path.isfile(path2mask):
                    imagemask = fits.open(path2mask)
                else:
                    print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                          'Calling calc_masks()...')
                    self.calc_masks(galaxy=galaxy, tile=tile, size=size,
                                    savemask=False, savefig=False)

                    mask_sexstars = True
                    maskstars = []
                    while mask_sexstars:
                        q1 = input('do you want to (UN)mask SExtractor stars? [y|r|n|q]: ')
                        if q1 == 'y':
                            newindx = input('type (space separated) the stars numbers do you WANT TO KEEP: ')
                            maskstars += [int(i) for i in newindx.split()]
                            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                                  'Current stars numbers are', maskstars)
                            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ',
                                  'Calling calc_masks()...')
                            self.calc_masks(galaxy=galaxy, tile=tile, size=size,
                                            savemask=False, savefig=False,
                                            maskstars=maskstars)
                            mask_sexstars = True
                        elif q1 == 'r':
                            maskstars = []
                        elif q1 == 'n':
                            maskstars = maskstars
                            mask_sexstars = False
                        elif q1 == 'q':
                            Warning('leaving!')
                            return
                        elif q1 == '':
                            mask_sexstars = True
                        else:
                            raise IOError('option not recognized')

                    imagemask = self.calc_masks(galaxy=galaxy, tile=tile, size=size,
                                                savemask=True, savefig=True,
                                                maskstars=maskstars)

                hdu3 = fits.ImageHDU(imagemask[1].data, imagemask[1].header)
                hdu3.header["EXTNAME"] = ("MASK", "Boolean mask of the galaxy")
                hdus.append(hdu3)
            thdu = fits.BinTableHDU(tab)
            hdus.append(thdu)
            thdu.header["EXTNAME"] = "METADATA"
            hdulist = fits.HDUList(hdus)
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Writing cube to', cubename)
            hdulist.writeto(cubename, overwrite=True)
            print('[%s]' % datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), ' - ', 'Cube successfully finished!')

if __name__ == "__main__":
    # print(initext)

    scubes = SCubes()

    scubes.work_dir = '/ssd/splus/S-cubes/'
    scubes.data_dir = '/home/herpich/Documents/pos-doc/t80s/Dropbox/splus-cubes/data/'
    scubes.zpcorr_dir = '/home/herpich/Documents/pos-doc/t80s/Dropbox/splus-cubes/data/zpcorr_idr3/'
    scubes.tiles_dir = os.path.join(scubes.work_dir, 'coadded')

    # # NGC1374
    # scubes.galaxies = np.array(['NGC1374'])
    # scubes.coords = [['03:35:16.598', '-35:13:34.50']]
    # scubes.tiles = ['SPLUS-s27s34']
    # scubes.sizes = np.array([600])
    # scubes.angsize = 180
    # specz = 0.004443

    # # NGC1399
    # scubes.galaxies = np.array(['NGC1399'])
    # scubes.coords = [['03:38:29.083', '-35:27:02.67']]
    # scubes.tiles = ['SPLUS-s27s34']
    # scubes.sizes = np.array([1800])
    # scubes.angsize = 540
    # specz = 0.004755

    # NGC1087
    # scubes.galaxies = np.array(['NGC1087'])
    # scubes.coords = [['02:46:25.15', '-00:29:55.45']]
    # scubes.tiles = ['STRIPE82-0059']
    # scubes.sizes = np.array([600])
    # scubes.angsize = 210
    # specz = 0.005070

    # # NGC1365
    # scubes.galaxies = np.array(['NGC1365'])
    # scubes.coords = [['03:33:36.458', '-36:08:26.37']]
    # scubes.tiles = ['SPLUS-s28s33']
    # scubes.sizes = np.array([1500])
    # scubes.angsize = 660
    # specz = 0.005476

    # NGC1326
    scubes.galaxies = np.array(['NGC1326'])
    scubes.coords = [['03:23:56.3657298384', '-36:27:52.322333040']]
    scubes.tiles = ['SPLUS-s28s32']
    scubes.sizes = np.array([800])
    scubes.angsize = 100
    specz = 0.004584
