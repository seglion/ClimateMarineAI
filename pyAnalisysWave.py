# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:01:23 2022

@author: mvigo
"""
#Directivas para el compilador

import pandas as pd;#Importamos la libreria pandas con la etiqueta pd
import numpy as np;
import scipy.stats as sci;
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import excel2img
import os 
from glob import glob1;
import datetime as dt;
import openpyxl.styles as st;
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule
from windrose import WindroseAxes
import seaborn as sns



class pyAnalysisWave():
    
    # Definimos el Constructor
    def __init__(self,Hs,Tp,Direc,fechas,Carpeta):
        """
        

        Parameters
        ----------
        Hs : Pandas.Series or ArrayList
            ALtura de Ola.
        Tp : Pandas.Series or ArrayList
            Periodo de Pico.
        Direc : Pandas.Series or ArrayList
            Direccion de Procedencia del Oleaje.
        fechas : ArrayList
            Lista con las fechas de los datos.

        Returns
        -------
        None.

        """
        
        d = {'Hs(m)' : Hs, 'Tp(seg)' : Tp, 'Dir(º)' :  Direc}
        self.data=pd.DataFrame(data=d, index = fechas);
        
        
        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14
        
        font = {'family' : 'Arial',
                'size'   : SMALL_SIZE,
                }
        
        plt.rc('font', **font)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        try:
            os.mkdir(os.getcwd() +'\\'+ Carpeta)
        except :
            print(' La Carpeta "' +Carpeta.upper() +'" ya estaba creada;Se sobreescribiran los Resultados')
            
            
        self.folder = os.getcwd() +'\\'+ Carpeta +'\\'
        
        self.bd = st.Side(style='thin', color="363842")
        
        self.head_style = st.NamedStyle(
        name="Head Style",
        font=st.Font(color='00FFFFFF', bold=True,name='Arial',sz=10),
        alignment=st.Alignment(horizontal='center'),
        fill = st.PatternFill('solid',fgColor='363842'),
        border=st.Border(left=self.bd, top=self.bd, right=self.bd, bottom=self.bd))
    
        # custom named style for the index
        self.body_style = st.NamedStyle(
        name="Body Style",
        font=st.Font(color='363842',name='Arial',sz=10),
        alignment=st.Alignment(horizontal='center'),
        number_format = '#,##0.00',
        border=st.Border(left=self.bd, top=self.bd, right=self.bd, bottom=self.bd))   
        
        self.percentile_rule = ColorScaleRule(
                    start_type='min', start_color='ffffff',
                    end_type='max', end_color='97C1FF')  # green-ish
        
        
        
    def AverageRegime(self,discretizacion=0.02):
        
        self.averageFolder = self.folder+'AverageRegime\\'  
        
        try:
            os.mkdir(self.averageFolder)
        except :
            print(' La Carpeta "' +self.averageFolder.upper() +'" ya estaba creada;Se sobreescribiran los Resultados')
            
            

        self.bins = np.arange(self.data['Hs(m)'].min(),self.data['Hs(m)'].max(),discretizacion)
        # Se selecciona la variable
        
        
        # Se obtienen las frecuencias de cada uno de los bins y los bordes se 
        # pasan a centros
        N = np.histogram(self.data['Hs(m)'], bins = self.bins)[0]
        centers = (self.bins[:-1] + self.bins[1:]) / 2
        
        
        Area = sum((centers[1] - centers[0]) * N)
        n = N/Area
        P1 = np.cumsum((centers[1]-centers[0])*n)
        P11 = P1[:-1] 
        y1 = centers[:-1]
        x1 = sci.norm.ppf(P11,loc=0,scale=1)  
        
        
        
        grid_kw = {'left':0.12, 'bottom':0.08, 'right':0.98, 'top':0.98}
        fig,ax = plt.subplots(figsize=(8,8),gridspec_kw=grid_kw)  
        #ax.scatter(x1,y1,s=20,c='red',alpha=0.5)
        ax.scatter(x1,np.log10(y1),s=20,c='red',alpha=0.5)  
        
        
        # ARREGLOS DEL GRÁFICO
        PP=np.array([0.05,0.2,0.5,0.8,0.95,0.99,0.999,0.9999])
        PP100=[]
        for i in PP:
            PP100.append(str(i*100)+'%')   
            
        yticks = np.round(np.arange(0.5,np.max(y1)+0.5,0.5),2)  
        ax.grid(which='both', linestyle=':',linewidth=1)
        ax.set_xlim(-2, sci.norm.ppf(0.999999,loc=0,scale=1))    
        ax.set_ylim(np.log10(0.5), np.log10(np.max(y1))+0.05*np.log10(np.max(y1)))
        ax.set_xticks(sci.norm.ppf(PP,loc=0,scale=1))
        ax.set_xticklabels(PP100)
        ax.set_yticks(np.log10(yticks))
        ax.set_yticklabels(yticks)
        ax.set_xlabel('Probabilidad de no excedencia',fontweight='bold');
        ax.set_ylabel('Hs(m)',fontweight='bold')
        
        p12=1-12/365/24
        percentiles = [0.5,0.9,0.95,0.99,0.999,p12]
        cols = ['Hs50%','Hs90%','Hs95%','Hs99%','Hs99.9%','Hs12']
        a = np.zeros((1,len(percentiles)))
        fig.savefig(self.averageFolder + 'Measured_Average_Regime.png',dpi=300)
        
        plt.close('all')        
        
        
        self.percMeasured = pd.DataFrame(a,columns=cols)
        
        #creacion de Tabla de Percentiles
        for i in range(len(percentiles)):   
            self.percMeasured[cols[i]] = np.interp(percentiles[i],sci.norm.cdf(x1,loc=0,scale=1),y1)

        output_filename=self.averageFolder +"Measured_Average_Regime_percentiles.xlsx"

        writer_args = {
        'path': output_filename,
        'mode': 'w',
        'engine': 'openpyxl'}

        with pd.ExcelWriter(**writer_args) as xlsx:
                sheet_name = "Percentiles"
                self.percMeasured.to_excel(xlsx, sheet_name)
                ws = xlsx.sheets[sheet_name]
                 
                # cell ranges
                index_column = 'A'
                value_cells = 'B2:{col}{row}'.format(
                    col=get_column_letter(ws.max_column),
                    row=ws.max_row)
                total_cells = 'B1:{col}{row}'.format(
                    col=get_column_letter(ws.max_column),
                    row=ws.max_row)
                title_row = '1'
             
                # index column width
                ws.column_dimensions.width = 8
                ws.column_dimensions[index_column].width = 1
        
                 
                # for general styling, one has to iterate over
                # all cells individually
                for row in ws[value_cells]:
                    for cell in row:
                        cell.number_format = '0.00'
                        cell.style = self.body_style
                # builtin or named styles can be applied by using
                # the style object or their name (shown below)
        
                 
                # style title row last, so that headline style
                # wins over index style in top-left cell A1
                for cell in ws[title_row]:
                    cell.style = self.head_style
                    
        # excel2img.export_img(output_filename,output_filename[:-4]+".png",sheet_name,total_cells)

    def Rose(self,maximo=6,discretizacion=0.5,cmap=cm.Spectral_r,Temporal=False):

        if Temporal: 
            self.ExtremeRoseFolder = self.folder+'Extreme_Wave_rose\\'  
            
            try:
                os.mkdir(self.ExtremeRoseFolder)
            except :
                print(' La Carpeta "' +self.ExtremeRoseFolder.upper() +'" ya estaba creada;Se sobreescribiran los Resultados')
            
            
            
            
            
            sp=np.arange(np.round(np.percentile(self.data['Hs(m)'],95)),maximo,discretizacion)
            
        else:
            
            self.AverageRoseFolder = self.folder+'Average_Wave_rose\\'  
            
            try:
                os.mkdir(self.AverageRoseFolder)
            except :
                print(' La Carpeta "' +self.AverageRoseFolder.upper() +'" ya estaba creada;Se sobreescribiran los Resultados')
            
            sp=np.arange(0,maximo,discretizacion)
       
        fig=plt.figure(figsize=(6,6))
        ax=WindroseAxes.from_ax(fig=fig)

        
        ax.grid(linestyle="dashed", zorder=0)
        ax.bar(self.data['Dir(º)'],self.data['Hs(m)'],normed=True, opening=1, edgecolor='black', linewidth=0.5, nsector=16, zorder= 3, cmap =cmap,bins=sp,)
        ax.set_legend(loc=(0.80, -0.05),title='Altura de Ola[m]',title_fontsize=10,fontsize=8)
        ax.set_xticklabels(['E', 'NE',  'N', 'NW', 'W', 'SW','S', 'SE'])
        ax.tick_params(axis='x', labelsize=6)
        col1 = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
        
        
        if Temporal:
            ax.axis([0.0, 6.283185307179586, 0.0, 50])
            ax.set_yticks(np.arange(0, 50, step=10))
            ax.set_yticklabels(np.arange(0, 50, step=10))
            ax.set_legend(loc=(0.80, -0.05),title='Altura de Ola \n Temporal[m]',title_fontsize=10,fontsize=8)
    
            fig.savefig(self.ExtremeRoseFolder+'ExtremeRose.png',dpi=150)
            output_filename=self.ExtremeRoseFolder+"Extreme_Rose_table_Sectores.xlsx"     
            output_filename2=self.ExtremeRoseFolder+"Extreme_Rose_table_Sectores.xlsx"
            
            
            
        else:
            ax.axis([0.0, 6.283185307179586, 0.0, 40])
            ax.set_yticks(np.arange(0, 40, step=5))
            ax.set_yticklabels(np.arange(0, 40, step=5))
            
            ax.set_legend(loc=(0.80, -0.05),title='Altura de Ola[m]',title_fontsize=10,fontsize=8)
            
            fig.savefig(self.AverageRoseFolder+'AverageRose.png',dpi=150)
            output_filename=self.AverageRoseFolder+"Average_Rose_table_Sectores.xlsx"     
            output_filename2=self.AverageRoseFolder+"Average_Rose_table_Sectores.xlsx"     



        writer_args = {
        'path': output_filename,
        'mode': 'w',
        'engine': 'openpyxl'}
        sectores = pd.DataFrame(ax._info['table'],columns=col1)
        sectores.index.name = 'Hs(m)'
       
        
       
        
       
        
       
        with pd.ExcelWriter(**writer_args) as xlsx:
                sheet_name = "Percentiles"
                sectores.to_excel(xlsx, sheet_name)
                ws = xlsx.sheets[sheet_name]
               
                title_row = '1'
                title_col = 'A'
                # index column width
               

               
                value_cells = 'B2:{col}{row}'.format(
                    col=get_column_letter(ws.max_column),
                    row=ws.max_row)
                
                total_cells = 'A1:{col}{row}'.format(
                    col=get_column_letter(ws.max_column),
                    row=ws.max_row)
                
                
                ws.conditional_formatting.add(value_cells, 
                                  self.percentile_rule)
                
                
                for cell in ws[title_row]:
                    cell.style = self.head_style
                    
                for cell in ws[title_col]:
                    cell.style = self.head_style
                
                
                
                for row in ws[value_cells]:
                    for cell in row:
                        cell.number_format = '0.00%'
                        cell.style = self.body_style 
                ws.column_dimensions.width = 5.29
        
        
        
        
        
        # excel2img.export_img(output_filename,output_filename[:-4]+".png",sheet_name,total_cells)
        
        
        
        
        
        
    def bivariate_distribution(self):
        self.ExtremeBivariateFolder = self.folder+'bivariate_distribution\\'
        try:
            os.mkdir(self.ExtremeBivariateFolder)
        except :
            print(' La Carpeta "' +self.ExtremeBivariateFolder.upper() +'" ya estaba creada;Se sobreescribiran los Resultados')
            
       
        # fig=plt.figure(figsize=(12  ,10  ))
        # ax=fig.add_axes([0.08, 0.08, 0.91, 0.91]) 
        # hb=plt.hexbin(self.data['Hs(m)'],self.data['Tp(seg)'],gridsize=20,mincnt=1,cmap='Spectral_r') 
        # cb = plt.colorbar()
        # cb.set_label('Probabilidad')
        # cb.set_ticks(np.linspace(hb.get_array().min(), hb.get_array().max(), 6))
        # cb.set_ticklabels(['%.4f'%x for x in np.linspace(hb.get_array().min()/hb.get_array().sum(), hb.get_array().max()/hb.get_array().sum(), 6)])
        # ax.grid(alpha=0.9)       
        # plt.show()
        # plt.savefig(self.ExtremeBivariateFolder + '\\DistHsTp_hex.png')        
        # return hb  
                   
        
        joint_kws=dict(gridsize=20)
        hexplot = sns.jointplot(x=self.data['Hs(m)'], y=self.data['Tp(seg)'] ,color = '#363842' ,kind="hex",space = 0, height = 10,ratio=5,joint_kws= joint_kws)
        plt.grid()
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
        cbar_ax = hexplot.fig.add_axes([.85, .25, .02, .4])  # x, y, width, height
    
        cbar = plt.colorbar(cax=cbar_ax)
        colorlabels = cbar.ax.get_yticks()
        colorlabels = np.round(colorlabels/len(self.data['Hs(m)']),4)
        cbar.ax.set_yticklabels(colorlabels)
        cbar.set_label('Frecuencia', fontsize = 12,fontweight='bold')
        #plt.show()
        
        plt.savefig(self.ExtremeBivariateFolder + '\\DistHsTp_hex.png')        
        pd.crosstab(index=self.data['Hs(m)'], columns = self.data['Tp(seg)'])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

if __name__ == '__main__':
   
    plt.close('all')
    ruta=r'K:\T2021\01 ESP\T2021-09-PORTOS_PROYECTOS_LOTE4\0405_LORBE\01_MODELOS\01_PYTHON\02_OLEAJE\01_INDEFINIDAS\02_CALIBRACION\\'

    data1=pd.read_csv(ruta+'10785_9602_3020040_WAVE_19580101085512_20221130085512_calibrado.dat',delimiter=',',header=0 )
    # fechas=[dt.datetime(int(trad[0][x]),int(trad[1][x]),int(trad[2][x]),int(trad[3][x])) for x in range(0,trad.shape[0])]
    # trad.index = fechas
    # trad.rename(columns={0:'year',1:'month',2:'Day',3:'Hour',4:'Altura Signif. del Oleaje(m)',6:'Periodo de Pico(s)',7:'Direcc. Media de Proced.(0=N,90=E)'},inplace=True)
    # trad[trad['Altura Signif. del Oleaje(m)']<0]=np.nan
    # data1=trad
    
    
    
    
    
    A = pyAnalysisWave(data1['Hs Calibrada'].to_numpy(),data1['Tp simar'].to_numpy(),data1['Dir simar'].to_numpy(),
                       data1.index,"3020040_-8.67_43.67")  
    A.AverageRegime(discretizacion=0.02)
    A.Rose()
    A.Rose(Temporal=True,cmap=cm.twilight,maximo=12,discretizacion=1)
    hb=A.bivariate_distribution()
    
    
    
    
    
    

    
    
    
    