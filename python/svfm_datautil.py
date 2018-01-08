from __future__ import print_function

import numpy as np
import xml.etree.ElementTree as xmletree



class SVFMDataReader:
    
    def __init__(self,prefix='simulation'):
        tree = xmletree.parse(prefix+'.info.xml')
        root = tree.getroot()
        general = root[0]
        
        # get general info
        print('# Reading info')
        for child in general:
            print(child.tag, child.attrib, child.text)
            if  (child.tag == 'nvort'):
                self.nvort = int(child.text)
            elif (child.tag == 'nimpt'):
                self.nimpt = int(child.text)
            elif (child.tag == 'dvort'):
                self.dvort = int(child.text)
            elif (child.tag == 'dimpt'):
                self.dimpt = int(child.text)
            elif (child.tag == 'velem'):
                self.velem = int(child.text)
            elif (child.tag == 'dt'):
                self.dt    = float(child.text)
            elif (child.tag == 'tend'):
                self.tend  = float(child.text)
            elif (child.tag == 'nom'):
                self.nom   = int(child.text)
        
        print('# Loading data (state vector)')
        self.state_vector = np.memmap(prefix+'_state.bin',dtype=np.float64)
        self.state_vector = np.reshape( self.state_vector,
                                       (self.nom,self.nvort*self.velem*self.dvort + self.nimpt*self.dimpt) )  # reshape by times
        
        
        if (self.nvort > 0):
            print('# Extracting vortices from state vector')
            self.xv = np.empty([self.nvort,self.nom,self.velem],dtype=np.float64,order='C')
            self.yv = np.empty([self.nvort,self.nom,self.velem],dtype=np.float64,order='C')
            
            vort_len = self.velem*self.dvort
            
            for it in range(self.nom):
                for jt in range(self.nvort):
                    self.xv[jt,it,:] = self.state_vector[it, vort_len*jt+0 : vort_len*(jt+1) : self.dvort]
                    self.yv[jt,it,:] = self.state_vector[it, vort_len*jt+1 : vort_len*(jt+1) : self.dvort]
        else:
            self.xv = None
            self.yv = None
        
        
        
        if (self.nimpt > 0):
            self.xN  = np.empty([self.nom,self.nimpt],dtype=np.float64,order='C')
            self.yN  = np.empty([self.nom,self.nimpt],dtype=np.float64,order='C')
            self.zN  = np.empty([self.nom,self.nimpt],dtype=np.float64,order='C')
            self.vxN = np.empty([self.nom,self.nimpt],dtype=np.float64,order='C')
            self.vyN = np.empty([self.nom,self.nimpt],dtype=np.float64,order='C')
            self.vzN = np.empty([self.nom,self.nimpt],dtype=np.float64,order='C')
            
            vorts_len = self.velem*self.dvort*self.nvort
            
            for it in range(self.nom):
                self.vxN[it,:] = self.state_vector[it, vorts_len+0::self.dimpt]
                self.vyN[it,:] = self.state_vector[it, vorts_len+1::self.dimpt]
                self.vzN[it,:] = self.state_vector[it, vorts_len+2::self.dimpt]
                self.xN[it,:]  = self.state_vector[it, vorts_len+3::self.dimpt]
                self.yN[it,:]  = self.state_vector[it, vorts_len+4::self.dimpt]
                self.zN[it,:]  = self.state_vector[it, vorts_len+5::self.dimpt]
        else:
            self.xN  = None
            self.yN  = None
            self.zN  = None
            self.vxN = None
            self.vyN = None
            self.vzN = None
        print('# Simulation data loaded.')
        
        if (self.nvort > 0):
            self.vind = np.memmap('velocity.bin',dtype=np.float64)
            self.vind = np.reshape( self.vind, [self.nom,self.nvort,3,self.velem] )
        
        
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
        