from __future__ import print_function

import numpy as np
import xml.etree.ElementTree as xmletree

import re

from tempfile import mkstemp
from shutil import move
from os import fdopen, remove


def replace_in_file(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


class XMLReader:
    
    def __init__(self,filename,verbose=False):
        if (filename[-9:] != '.info.xml'):
            filename = filename + '.info.xml'
        self.tree = xmletree.parse(filename)
        self.root = self.tree.getroot()
        
        if (verbose is True):
            for r in self.root:
                print(r)
    
    def show_xml(self):
        # get general info
        print('# Reading info')
        for category in self.root:
            print(category)
            for child in category:
                print('    ',child.tag, child.attrib, child.text)
        print()


class SVFMDataReader(XMLReader):
    
    match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\+?-?\ *[0-9]+)?')
    
    prefix  = None
    eta     = None
    vext_x  = None
    vext_y  = None
    nnucul  = None
    zmin    = None
    zmax    = None
    dz      = None
    N       = None
    nom     = None
    tmax    = None
    vort    = None
    nucl    = None
    t       = None
    dt      = None
    alatt   = None
    
    def __init__(self,filename,verbose=False):
        XMLReader.__init__(self,filename,verbose=verbose)
        
        self.parameters = self.root[0]
        
        if verbose is True:
            print(filename)
            print(re.findall(self.match_number, filename))
            print([float(x) for x in re.findall(self.match_number, filename)])
        
        self.prefix  =       self.parameters.findall('prefix')[0].text
        self.eta     = float(self.parameters.findall('eta')[0].text   )
        self.vext_x  = float(self.parameters.findall('vext_x')[0].text)
        self.vext_y  = float(self.parameters.findall('vext_y')[0].text)
        self.mn      = float(self.parameters.findall('neutron_mass_MeV')[0].text)
        self.kappa   = float(self.parameters.findall('kappa')[0].text )
        self.kF_n    = float(self.parameters.findall('kF_n')[0].text  )
        self.delta   = float(self.parameters.findall('delta')[0].text )
        self.rho_n   = float(self.parameters.findall('rho_n')[0].text )
        self.Tv      = float(re.findall(self.match_number, filename)[1]   )
        self.xi      = float(self.parameters.findall('xi_BCS')[0].text    )
        self.nnucul  = float(self.parameters.findall('num_nuclei')[0].text)
        self.zmin    = float(self.parameters.findall('zmin')[0].text  )
        self.zmax    = float(self.parameters.findall('zmax')[0].text  )
        self.dz      = float(self.parameters.findall('dz')[0].text    )
        self.N       =   int(self.parameters.findall('N')[0].text     )
        self.nom     =   int(self.parameters.findall('nom')[0].text   )
        self.tmax    = float(self.parameters.findall('tmax')[0].text  )
        self.alatt   = float(self.parameters.findall('lattice_const')[0].text  )
        
        self.dt = self.tmax / float(self.nom)
        
        if verbose is True:
            print(self.prefix)
        
        
        
        
        # load data
        if (filename[-9:] == '.info.xml'):
            filename = filename[:-9]
        self.vort = np.memmap(filename+'.vort.bin',dtype=np.float64); vsize = self.vort.size;
        self.vort = np.reshape(self.vort,[vsize//(2*self.N),2*self.N])
        if (self.nom != vsize//(2*self.N) ):
            self.nom  = vsize//(2*self.N)
        
        self.t = np.linspace(0.,self.tmax,self.nom)
        
        
        self.nucl = np.memmap(filename+'.nucl.bin',dtype=np.float64); nsize = self.nucl.size;
        self.nucl = np.reshape(self.nucl,[nsize//(3*self.nnucul),self.nnucul,3])
        
        if verbose is True:
            print(nsize,nsize//3,nsize//self.nom,nsize//self.nnucul)
            print()



class SVFMDataWriter:
    
    # constructor
    def __init__(self,filename=None):
        """
        @param filename       -  name of a file where parameters will be written to, extension .info.xml will be added automatically
        """
        self.filename = None
        if filename is not None:
            self.filename = filename + '.info.xml'
        self.root = xmletree.Element("info")
        self.params = xmletree.SubElement(self.root,"parameters")
    
    # destructor
    #def __del__(self)
    #    self.tree = ET.ElementTree(self.root)
    #    self.tree.write("filename.xml")
    
    def add_parameter(self,name,value):
        xmletree.SubElement(self.params,name).text = str(value)
    
    def set_filename(self,filename):
        self.filename = filename + '.info.xml'
    
    def save_info(self,filename=None):
        if filename is not None:
            self.filename = filename + '.info.xml'
        self.tree = xmletree.ElementTree(self.root)
        self.tree.write(self.filename)
        