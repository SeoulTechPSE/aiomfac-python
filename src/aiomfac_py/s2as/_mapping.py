# Vendored from https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC (commit 88a2bffde1d375f8cb86a6e47ead6c3f0dad1833),
# GPL-3.0-or-later. See ../s2as/LICENSE_S2AS and __init__.py for integration notes.

# AZ: this module (used by SMILES_to_AIOMFAC_input.py) contains the main mapping functions 
# from SMILES to AIOMFAC subgroups (S2AS), covering both the regular cases with handling 
# of specified exceptions for imperfect matching of substructures by AIOMFAC subgroups, 
# as well as the separate treatment of so-called pure alcohols or polyols.

from indigo import *
from indigo import renderer as Renderer
from ._smarts import SMARTS_AIOMFAC


#------------------------------------------------------------------------------------------------------
# section to define specific lists for pure alcohol/polyol matching purposes:

# alkylPlusOH = list with the CHn groups attached to hydroxyl:
alkylOH = ['[CH2]([OX2H1])[OX2H1]', '[CH1]([OX2H1])[OX2H1]', '[CH0]([OX2H1])[OX2H1]', 
           '[CH3][OX2H1]', '[CH2][OX2H1]', '[CH1][OX2H1]', '[CH0][OX2H1]']
subgCHnOH = [1120, 1121, 1122, 149, 150, 151, 152]    #(there is also AIOMFAC subgroup 153 = -OH present)

# alkylTailEnd = list of SMARTS for detecting the two end carbons of a hydrophobic tail in alcohols:
alkylTailEnd = ['[CH3][CH2]', '[CH3][CH1]', '[CH3][CH0]']
subgTailEnd  = [146, 147, 148]      #each of these matches will also include subgroup no. 145 for -CH3; 

# alkylGroups = list of SMARTS for general alkyl subgroups for use with alcohols:
alkylG = ['[CH3]', '[CH2]', '[CH1]', '[CH0]']
subgTail = [145, 146, 147, 148]     #list of special alkyl in a hydrophobic tail AIOMFAC subgroup numbers;
subgAlc  = [141, 142, 143, 144]     #list of other "in alcohol" alkyl AIOMFAC subgroup numbers;

#------------------------------------------------------------------------------------------------------


# ====== Main Function ===========================================================

# Define a function that works on the main task of matching SMARTS patterns 
# and returning the subgroup array;
def detAIOMFACsubgs(ind, molec, is_PureAlc, debugging_verbose, debugRenderedStructures, 
                    image_file_type, showRenderedStructures, igatoms_index, 
                    neigh_atom_list, highLAtomsList):

   indigo = Indigo()
   renderer = Renderer.IndigoRenderer(indigo)

   sequence_count = 0
   subgs = [0 for j in range(0,173)] 
   count_SMARTS_exceptions = 0

   igatoms_index.clear()  
   structure = indigo.loadMolecule(molec) 
   if structure.canonicalSmiles() != '':
      matcher = indigo.substructureMatcher(structure)
   else:
      info = -99
      return subgs, info   #exceptional return point;

   if debugging_verbose:
      print('')
      print('Input SMILES of molecule: ' + structure.canonicalSmiles() + ' ind ' , ind)
   molecatomcount = structure.countAtoms()

   if (not is_PureAlc):
      #======
      # (1) iterate over all general AIOMFAC SMARTS patterns and count number 
      #     of matches in current molecule:
      alkyldeduct = 0
      CH3deduct = 0
      for row in range(len(SMARTS_AIOMFAC[:])):
         sub,item = SMARTS_AIOMFAC[row][0:]
         tsmarts = indigo.loadSmarts(item)    
         match1 = matcher.match(tsmarts)
         has_SMARTS_match = False
         while match1 != None: 
            #...check for exception cases and count matched subgroup:
            if sub == 1550:
               subgs[155] = subgs[155] +1
               alkyldeduct += 1     #counter for how many alkyl groups (detected later) should be deducted due to an exception case match here;
            
            elif sub == 1580:
               subgs[158] = subgs[158] +1
               alkyldeduct += 1
            
            elif sub == 1700:
               subgs[170] = subgs[170] +1
               alkyldeduct += 1

            elif sub == 1710:
               subgs[171] = subgs[171] +1
               alkyldeduct += 2 

            elif sub == 2002:
               subgs[20] = subgs[20] +1
               alkyldeduct += 1 

            elif sub == 2022:
               subgs[20] = subgs[20] +1
               subgs[22] = subgs[22] +1

            elif sub == 2220:
               subgs[20] = subgs[20] +1
               subgs[22] = subgs[22] +1
               alkyldeduct += 1

            elif sub == 2224:
               subgs[20] = subgs[20] +1
               subgs[24] = subgs[24] +1
               CH3deduct += 1

            elif sub == 2502 or (sub == 2501):
               subgs[25] = subgs[25] +1
               alkyldeduct += 1

            elif sub == 926:
                 subgs[9] += 3
                 subgs[26] += 1
                 
            elif (sub == 1026) or (sub == 1027):
                 subgs[9] += 2
                 subgs[10] += 1
                 subgs[26] += 1
            
             # geminal
            elif sub == 1120:
                 subgs[150] += 1
                 subgs[153] += 2
            
            elif sub == 1121:
                 subgs[151] += 1
                 subgs[153] += 2
           
            elif sub == 1122:
                 subgs[152] += 1
                 subgs[153] += 2
            
            elif sub == 560:
                 subgs[56] += 1
                 alkyldeduct += 1

            else: #regular case
               subgs[sub] = subgs[sub] +1
            #...

            if sub > 300:
               count_SMARTS_exceptions += 1

            for atom in tsmarts.iterateAtoms():
               if match1.mapAtom(atom) != None:
                  atom_index = match1.mapAtom(atom).index()
                  igatoms_index.append(atom_index)
                  atom_to_ignore = structure.getAtom(atom_index)
                  if debugRenderedStructures:
                     has_SMARTS_match = True
                     atom_to_ignore.highlight()
                  matcher.ignoreAtom(atom_to_ignore) 
            match1 = matcher.match(tsmarts)              #load the matcher again since some atoms from this molecule may have been ignored;

         #---- save an image of this component with highlighted subgroup 
         #     (for debugging, otherwise not needed):
         if debugRenderedStructures and has_SMARTS_match:
            sequence_count += 1
            #produce output file with highlighted subgroup matches, all else in black:
            indigo.setOption("render-output-format", image_file_type)
            indigo.setOption("render-margins", "5, 5")
            indigo.setOption("render-coloring", "False")
            indigo.setOption("render-background-color", "1.0, 1.0, 1.0")
            indigo.setOption("render-atom-ids-visible", "False")           #use True to show atom id numbers of structure
            indigo.setOption("render-label-mode", "hetero")
            indigo.setOption("render-highlight-color-enabled", "True")
            indigo.setOption("render-highlight-color", "1.0, 0.4, 0.0")    # orange-red for highlighted atoms
            st = str(ind).zfill(3) + "_seq" +str(sequence_count).zfill(3)
            csub = str(sub).zfill(3)
            crow = str(row).zfill(2)  #to mark the SMARTS priority position
            outpathfile = "OutputFiles/Structure_" +st + "_SMARTS_" +crow +"_subgr_" +csub +str("." + image_file_type)
            renderer.renderToFile(structure, outpathfile)
            #now unhighlight the atoms from the current match:
            for atom in structure.iterateAtoms():
               atom.unhighlight()
         #---- end of save an image

      #check whether CH3 alkyl groups should be deducted (e.g. due to imperfect matching of per-ester group):
      if CH3deduct > 0:
         deduct_is_possible = True
         while CH3deduct > 0 and deduct_is_possible:
            if subgs[1] > 0:
               subgs[1] -= 1
               CH3deduct -= 1
            else:
               deduct_is_possible = False
               alkyldeduct += CH3deduct

      #check whether any alkyl groups should be deducted because of imperfect subgroup matching by other groups;
      #(doing this will ensure that the number of C atoms is at least correctly accounted for):
      if alkyldeduct > 0:
         deduct_is_possible = True
         while alkyldeduct > 0 and deduct_is_possible:
            if subgs[3] > 0:
               subgs[3] -= 1
               alkyldeduct -= 1

            elif subgs[2] > 0:
               subgs[2] -= 1
               alkyldeduct -= 1

            elif subgs[4] > 0:
               subgs[4] -= 1
               alkyldeduct -= 1

            elif subgs[1] > 0:
               subgs[1] -= 1
               alkyldeduct -= 1

            else:
               deduct_is_possible = False
      #=======

   else:  
      #--> it is a pure aliphatic alcohol/polyol, so perform special pattern searches (steps 1 - 3);
      #--------
      # step (1) determine the CHn with OH groups and the OH groups themselves;
      for i,item in enumerate(alkylOH): 
         tsmarts = indigo.loadSmarts(item)    
         matchCHnOH = matcher.match(tsmarts)
         has_SMARTS_match = False

         while matchCHnOH != None:
            sub = subgCHnOH[i]

            # Geminal diol cases
            if sub == 1120:
               subgs[150] += 1
               subgs[153] += 2
             
            elif sub == 1121:
               subgs[151] += 1
               subgs[153] += 2
            
            elif sub == 1122:
               subgs[152] += 1
               subgs[153] += 2

            else:
               subgs[sub] += 1      #count the matched CHn (with OH) subgroup; 
               subgs[153] += 1      #also count the matched OH subgroup;

            #...
            for atom in tsmarts.iterateAtoms():
               atom_to_ignore = matchCHnOH.mapAtom(atom)
               igatoms_index.append(atom_to_ignore.index())
               matcher.ignoreAtom(atom_to_ignore)

               if debugRenderedStructures:
                  has_SMARTS_match = True
                  atom_to_ignore.highlight()
            #...
            matchCHnOH = matcher.match(tsmarts)          #load the matcher again since some atoms from this molecule may have been ignored;

         #---- save an image of this component with highlighted subgroup 
         #     (for debugging, otherwise not needed):
         if debugRenderedStructures and has_SMARTS_match:
            sequence_count += 1
            #produce output file with highlighted subgroup matches, all else in black:
            indigo.setOption("render-output-format", image_file_type)
            indigo.setOption("render-margins", "5, 5")
            indigo.setOption("render-coloring", "False")
            indigo.setOption("render-background-color", "1.0, 1.0, 1.0")
            indigo.setOption("render-atom-ids-visible", "False")
            indigo.setOption("render-label-mode", "hetero")
            indigo.setOption("render-highlight-color-enabled", "True")
            indigo.setOption("render-highlight-color", "0.117, 0.565, 1.0")    # dodgerblue color for highlighted CHn--OH atoms
            st = str(ind).zfill(3) + "_seq" +str(sequence_count).zfill(3)
            csub = str(sub).zfill(3)
            crow = str("alkylOH")  #to mark the SMARTS priority position
            outpathfile = "OutputFiles/Structure_" +st + "_SMARTS_" +crow +"_subgr_" +csub +str("." + image_file_type)
            renderer.renderToFile(structure, outpathfile)
            #now unhighlight the atoms from the current match (here were only highlighted for image rendering):
            for atom in structure.iterateAtoms():
               atom.unhighlight()
         #---- end of save an image

      #--------
      # step (2) determine hydrophobic chains that terminate in -CHn-CH3  (n = 0,1,2); 
      #    [use Indigo highlight and isHighlighted functions to mark CHn atoms that 
      #    neighbor tail groups for testing in code to indicate whether chain of 
      #    hydrophobic tail continues.]
      neigh_atom_list.clear()
      highLAtomsList.clear()

      # (2a) determine the hydrophobic tail end groups (if present) and highlight neighboring atoms (used later as tail marker);
      for i,item in enumerate(alkylTailEnd): 
         tsmarts = indigo.loadSmarts(item)    
         matchCHntail = matcher.match(tsmarts)
         has_SMARTS_match = False

         while matchCHntail != None:
            sub = subgTailEnd[i]
            subgs[sub] = subgs[sub] +1       #register the matched CHn (n = 0,1,2) subgroup; 
            subgs[145] = subgs[145] +1       #also register the matched CH3 in hydrophobic tail subgroup;
            #...
            for atom in tsmarts.iterateAtoms():
               targetAtom = matchCHntail.mapAtom(atom)
               if debugRenderedStructures: 
                  has_SMARTS_match = True
                  targetAtom.highlight() 
                  
               # first highlight the neighboring atom of the matched group, then ignore the matched atoms:
               igatoms_index.append(targetAtom.index())
               for neiatom in targetAtom.iterateNeighbors():
                  neiatom.highlight()
                  neigh_atom_list.append(neiatom.index())

               matcher.ignoreAtom(targetAtom)
            #...
            matchCHntail = matcher.match(tsmarts)           #load the matcher again since some atoms from this molecule may have been ignored;

         #---- save an image of this component with highlighted subgroup 
         #     (for debugging/visualization, otherwise not needed):
         if debugRenderedStructures and has_SMARTS_match:
            sequence_count += 1

            #temporarily unhighlight neighboring atoms that were not yet matched to specific 
            #subgroups (just for image below)
            for atom in structure.iterateAtoms():
               if atom.index() not in igatoms_index:
                  if atom.index() in neigh_atom_list: 
                     atom.unhighlight()
            #produce output file with highlighted subgroup matches, all else in black:
            indigo.setOption("render-output-format", image_file_type)
            indigo.setOption("render-margins", "5, 5")
            indigo.setOption("render-coloring", "False")
            indigo.setOption("render-background-color", "1.0, 1.0, 1.0")
            indigo.setOption("render-atom-ids-visible", "False")
            indigo.setOption("render-label-mode", "hetero")
            indigo.setOption("render-highlight-color-enabled", "True")
            indigo.setOption("render-highlight-color", "1.0, 0.7, 0.0")    # yellow-orange for highlighted tail atoms
            st = str(ind).zfill(3) + "_seq" +str(sequence_count).zfill(3)
            csub = str(sub).zfill(3) + str("_145")
            crow = str("alkylTailEnd")  #to mark the SMARTS priority position
            outpathfile = "OutputFiles/Structure_" +st +"_SMARTS_" +crow +"_subgr_" +csub +str("." + image_file_type)
            renderer.renderToFile(structure, outpathfile)          
            for atom in structure.iterateAtoms():
               if atom.index() not in igatoms_index:
                  if atom.index() in neigh_atom_list:    
                     atom.highlight()                 # re-highlight designated tail-neighboring atoms that were temporarily unhighlighted
               else:
                  atom.unhighlight()                  # unhighlight the atoms from the current match (here were only highlighted for purpose of image rendering):
         #---- end of save an image

      # check on the indices of highlighted atoms in the molecule:
      for atom in structure.iterateAtoms():
         if atom.index() in igatoms_index: 
            atom.unhighlight()                        # exclude ignored atoms from being highlighted (here only show highlighted neighbour atoms that have not been mapped to subgroups);
         elif atom.isHighlighted():
            highLAtomsList.append(atom.index())

      # (2b) highlight all other hydrophobic tail atoms and associate them with specific alkyl tail subgroups:
      #      Note: an alkyl group is part of a hydrophobic tail if and only if it neighbors at least one other alkyl group in a hydrophobic tail 
      #      and is not connected to a OH group (those have been determined already in step 1);
      temporaryIgnoreL = []
      rendered_highlight_list = []
      mcounter = 0
      tsmarts = indigo.loadSmarts('[CX4]')                  #this SMARTS will hit any alkyl carbon that has not been ignored;
      matchCHntail = matcher.match(tsmarts)
      while matchCHntail != None:
         #...
         has_SMARTS_match = False
         for atom in tsmarts.iterateAtoms():
            targetAtom = matchCHntail.mapAtom(atom)
            if targetAtom.isHighlighted():                  #select highlighted atoms only!
               #-.-
               for neiatom in targetAtom.iterateNeighbors():
                  neiatom_index = neiatom.index()

                  if neiatom_index not in igatoms_index and (not neiatom.isHighlighted()):
                     neiatom.highlight()
                     highLAtomsList.append(neiatom_index)

                     if neiatom_index in temporaryIgnoreL:
                        temporaryIgnoreL.remove(neiatom_index)
                        matcher.unignoreAtom(neiatom)       #unignore this atom as it is now highlighted and a tail atom;
               #-.-
               #determine the specific tail alkyl subgroups and then ignore the atom:
               kH = targetAtom.countHydrogens()
               sub = 148 - kH                               #here to hit the correct subgroup ID for hydrophobic tail CHn groups
               subgs[sub] = subgs[sub] +1
               igatoms_index.append(targetAtom.index())
               matcher.ignoreAtom(targetAtom)               #ignore current atom to allow matcher to progress to next hit;
               if debugRenderedStructures:                  #just for debugging image plots;
                  has_SMARTS_match = True
                  mcounter += 1

            else:
               temporaryIgnoreL.append(targetAtom.index())  #add atom index of matched atom to a temporary ignore list; later unignored for final matching;
               matcher.ignoreAtom(targetAtom)               #ignore current atom to allow matcher to progress to next hit;
         #...
         matchCHntail = matcher.match(tsmarts)              #load the matcher again since some atoms from this molecule may have been ignored;

         #---- save an image of this component with highlighted subgroup 
         #     (for debugging/visualization, otherwise not needed):
         if debugRenderedStructures and has_SMARTS_match:
            sequence_count += 1
            # temporarily unhighlight neighboring atoms that were not yet matched 
            # to specific subgroups (just for image below)
            for atom in structure.iterateAtoms():
               if atom.index() not in igatoms_index:
                  if atom.index() in highLAtomsList: 
                     atom.unhighlight()
               if atom.isHighlighted() and atom.index() in rendered_highlight_list:
                  atom.unhighlight()
            # produce output file with highlighted subgroup matches, all else in black:
            indigo.setOption("render-output-format", image_file_type)
            indigo.setOption("render-margins", "5, 5")
            indigo.setOption("render-coloring", "False")
            indigo.setOption("render-background-color", "1.0, 1.0, 1.0")
            indigo.setOption("render-atom-ids-visible", "False")
            indigo.setOption("render-label-mode", "hetero")
            indigo.setOption("render-highlight-color-enabled", "True")
            indigo.setOption("render-highlight-color", "1.0, 0.7, 0.0")    # yellow-orange for highlighted tail atoms
            st = str(ind).zfill(3) + "_seq" +str(sequence_count).zfill(3)
            csub = str(sub).zfill(3)
            crow = str("alkylTailMid" +str(mcounter).zfill(3))
            outpathfile = "OutputFiles/Structure_" +st + "_SMARTS_" +crow +"_subgr_" +csub +str("." + image_file_type)
            renderer.renderToFile(structure, outpathfile)    
            for atom in structure.iterateAtoms():
               if atom.isHighlighted():
                  rendered_highlight_list.append(atom.index())    #keep track of already highlighted atoms in image (to unhighlight in next image) 
               if atom.index() not in igatoms_index:
                  if atom.index() in highLAtomsList:    
                     atom.highlight()                 # re-highlight designated tail-neighboring atoms that were temporarily unhighlighted
               #else:
               #   atom.unhighlight()                  # unhighlight the atoms from the current match (here were only highlighted for image rendering):
         #---- end of save an image
      
      # unignore temporarily ignored atoms:
      for atom in structure.iterateAtoms():
         atom.unhighlight()  #unhighlight all atoms from step (2);
         if atom.index() in temporaryIgnoreL: 
            matcher.unignoreAtom(atom) 
            temporaryIgnoreL.remove(atom.index())

      #--------
      # step (3) assign alkyl within alcohols type to all remaining CHn groups.
      has_SMARTS_match = False
      for i,item in enumerate(alkylG): 
         tsmarts = indigo.loadSmarts(item)    
         matchCHnAlc = matcher.match(tsmarts)

         while matchCHnAlc != None:
            sub = subgAlc[i]
            subgs[sub] = subgs[sub] +1                   #register the matched CHn (n = 0,1,2) subgroup; 
            #...
            for atom in tsmarts.iterateAtoms():
               targetAtom = matchCHnAlc.mapAtom(atom)
               igatoms_index.append(targetAtom.index())
               matcher.ignoreAtom(targetAtom)
               if debugRenderedStructures:               #just for debugging image plots;
                  has_SMARTS_match = True
                  targetAtom.highlight()
            #...
            matchCHnAlc = matcher.match(tsmarts)         #load the matcher again since some atoms from this molecule may have been ignored;

      #---- save an image of this component with highlighted subgroups 
      #     (for debugging/visualization, otherwise not needed):
      if debugRenderedStructures and has_SMARTS_match:
         sequence_count += 1
         #produce output file with highlighted subgroup matches, all else in black:
         indigo.setOption("render-output-format", image_file_type)
         indigo.setOption("render-margins", "5, 5")
         indigo.setOption("render-coloring", "False")
         indigo.setOption("render-background-color", "1.0, 1.0, 1.0")
         indigo.setOption("render-atom-ids-visible", "False")
         indigo.setOption("render-label-mode", "hetero")
         indigo.setOption("render-highlight-color-enabled", "True")
         indigo.setOption("render-highlight-color", "1.0, 0.2, 0.4")    # pink-red color for highlighted CHn in alcohol atoms
         st = str(ind).zfill(3) + "_seq" +str(sequence_count).zfill(3)
         csub = str(sub).zfill(3)
         crow = str("alkylAlcs")  #to mark the SMARTS priority position
         outpathfile = "OutputFiles/Structure_" +st + "_SMARTS_" +crow +"_subgr_" +csub +str("." + image_file_type)
         renderer.renderToFile(structure, outpathfile)    
      #---- end of save an image

   #---- save an image of this component's overall structure (for visualization)
   if showRenderedStructures:
      for atom in structure.iterateAtoms():
         atom.unhighlight()
      #produce new output file:
      indigo.setOption("render-output-format", image_file_type)
      indigo.setOption("render-margins", "5, 5")
      indigo.setOption("render-coloring", "False")
      indigo.setOption("render-background-color", "1.0, 1.0, 1.0")
      indigo.setOption("render-atom-ids-visible", "True")
      indigo.setOption("render-label-mode", "hetero")
      indigo.setOption("render-highlight-color-enabled", "True")
      indigo.setOption("render-highlight-color", "1.0, 0.55, 0.0")    # orange for highlighted atoms
      st = str(ind).zfill(3)
      outpathfile = "OutputFiles/Structure_" +st + str("." +image_file_type)
      renderer.renderToFile(structure, outpathfile) 
   #---- end of save an image of overall structure

   if debugging_verbose:
      print('Total number of atoms in input molecule are : ', molecatomcount)
      print('Total number of atoms that have been ignored after successful matching are : ', len(igatoms_index))
      if subgs[0] > 0:
         print('WARNING: not all atoms were assigned to proper AIOMFAC subgroups!')
         print('The count of subgroup-0 atoms is non-zero (though it should always be 0).')
         input()
   #check whether issues occurred that may need attention:
   ii = molecatomcount - len(igatoms_index)
   if ii != 0:
      print('WARNING: not all atoms were correctly matched to AIOMFAC subgroups for the input SMILES at ind ' + str(ind).zfill(2))
      print('Number of non-H atoms not matched to any AIOMFAC subgroup: ' + str(ii).zfill(2))

      i = ii

   return subgs, ii, count_SMARTS_exceptions
# === end of function ===================================================================
   
# explicitly clean up indigo resources before exit
del renderer
del indigo