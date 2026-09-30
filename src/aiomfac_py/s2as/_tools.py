# Vendored from https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC (commit 88a2bffde1d375f8cb86a6e47ead6c3f0dad1833),
# GPL-3.0-or-later. See ../s2as/LICENSE_S2AS and __init__.py for integration notes.
########################################################################### 
# 
#  Module 'ModSmilesTools'
#
#  :: Purpose ::
#  Module containing custom-made functions for SMARTS substructure searches 
#  and specific checks for given SMILES using the Indigo toolkit.
#
#  :: Authors & Copyright ::                                         
#   Andreas Zuend, Dalrin Ampritta Amaladhasan,
#   Dept. Atmospheric and Oceanic Sciences, McGill University    
#   
#   :: List of functions contained in this module:
#   ----------------------------------------------
#   -  PureAlcoholCheck                                 
#   -  verifySMILES                                   
#   -  CompDataToStringList                              
#   -  writeAIOMFACfile                               
#
###########################################################################

from indigo import *
from indigo import renderer as renderer

indigo = Indigo()
renderer = renderer.IndigoRenderer(indigo)

#------------------------------------------------------------------------------------------
# Function to check whether an input SMILES represents a pure alcohol 
# (with special alkyl subgroups in AIOMFAC) or not. Returns True or False.
def PureAlcoholCheck(inpsmiles): 
   containsOH = '[C][OX2H1]'              #SMARTS to probe whether a molecule contains an aliphatic -OH group (while avoiding a match with hydroperoxide group)
   containsNonOH = '[*;OH0,S,N,a]'        #SMARTS to probe whether other functional groups are present that make the molecule not a pure alcohol, incl. C=C double and triple bonds;
   containsDblorTriBond = '[C]=,#[C]'
   is_pure_alc = False                    #default value, potentially overwritten below 

   molec = indigo.loadMolecule(inpsmiles)
   if molec.canonicalSmiles() != '':
      matcher = indigo.substructureMatcher(molec)
      matchSMARTS = matcher.match(indigo.loadSmarts(containsOH))           
      if matchSMARTS != None:
         matchThis = matcher.match(indigo.loadSmarts(containsNonOH))
         if matchThis == None:      #--> no non-OH functionality containing oxygen, S, N, or aromatic atoms;
            matchThis2 = matcher.match(indigo.loadSmarts(containsDblorTriBond))
            if matchThis2 == None:  #--> no double or triple bonds present;
               is_pure_alc = True

   #print('')
   #print('input SMILES of molecule is: ' + molec.canonicalSmiles())
   #print('the molecule is a "pure" alcohol (T/F?): ' + str(is_pure_alc))
   return is_pure_alc
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
# Function to verify whether an input SMILES represents a valid molecule; returns True or False.
def verifySMILES(inpsmiles):
   is_valid = False
   errstring = ''
   molec = indigo.loadMolecule(inpsmiles)
   check1 = molec.checkBadValence()
   if check1 == '':
      check2 = molec.checkAmbiguousH()
      if check2 == '':
         is_valid = True
   else:
      check2 = ''

   errstring = check1 + check2
   return is_valid, errstring
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
# Function to write subgroup information data for each (organic) component to a list of strings
# output is tab (\t) delimited where applicable;
def CompDataToStringList(SMILESlist, cpsubs, cp0inc):
   outpDat = []
   for ind,molec in enumerate(SMILESlist):
      molectext = str(molec).replace('\n','')
      outpDat.append('component no.:\t' + str(ind+cp0inc).zfill(2))
      outpDat.append('component name:\t' + "'" + molectext + "'")     #component name enclosed in ''
      for sub in range(1,173):               #output the non-zero amount subgroups of this component
         qty = cpsubs[ind][sub]
         if qty > 0:
            outpDat.append('subgroup no., qty:\t' + str(sub).zfill(3) + ',\t' + str(qty).zfill(2))
      outpDat.append('----')

   return outpDat
#------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
# Function to write a valid AIOMFAC-web style input file characterizing the system components.
def writeAIOMFACfile(fname, outpath, SMILESlist, cpsubs, waterAsComp01):

   txtdataL = []     #the string list containing the output data;
   txtdataL.append('Input file for AIOMFAC-web model')
   txtdataL.append('')
   txtdataL.append('mixture components:')
   txtdataL.append('----')
   if waterAsComp01: #add water as component 1;
      cp0inc = 2     #increment in component numbering index used for components 2+
      txtdataL.append('component no.:\t' + str(1).zfill(2))
      txtdataL.append('component name:\t' + "'"'Water'"'") 
      txtdataL.append('subgroup no., qty:\t' + str(16).zfill(3) + ',\t' + str(1).zfill(2))
      txtdataL.append('----')
   else:
      cp0inc = 1
   #for all other (organic) components:
   cpdata = CompDataToStringList(SMILESlist, cpsubs, cp0inc)
   for row in range(len(cpdata)):
      txtdataL.append(cpdata[row])

   #add dummy composition data; (it will typically not be used for actual calculations, but makes the file a valid AIOMFAC-web input file)
   txtdataL.append('++++')
   txtdataL.append('mixture composition and temperature:')
   txtdataL.append('mass fraction? 0')
   txtdataL.append('mole fraction? 1')
   txtdataL.append('----')
   # add a generated a string of all the component numbers; cp01, cp02, cp03, ...
   st = 'point, T_K' 
   for i,nmolec in enumerate(SMILESlist, start=2):
      st = st + (', cp' + str(i).zfill(2))
   txtdataL.append(st)
   jj = list(float(1.0E-12) for i,nmolec in enumerate(SMILESlist))
   st = '1, 298.15, ' + str(jj)[1:-1]
   txtdataL.append(st)
   txtdataL.append('====')

   #output to text file:
   outpfile = outpath+'/' + fname
   fileA = open(outpfile, 'w')
   for item in txtdataL:
      fileA.write("{}\n".format(item))
   fileA.close()
   return
#------------------------------------------------------------------------------------------