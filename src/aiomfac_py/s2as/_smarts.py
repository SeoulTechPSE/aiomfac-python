# Vendored from https://github.com/andizuend/S2AS__SMILES_to_AIOMFAC (commit 88a2bffde1d375f8cb86a6e47ead6c3f0dad1833),
# GPL-3.0-or-later. See ../s2as/LICENSE_S2AS and __init__.py for integration notes.
#
# This file contains the general SMARTS query list in order of priority for mapping of AIOMFAC subgroups.
#
# :: REMARKS ::
# Highest priority groups are at the beginning (top) of the list. 
# Note that the relative positions of individual SMARTS strings absolutely matter in this list for matches to be unique (since one assumes that other subgroups of higher priority where already matched).
# The special alcohol/polyol UNIFAC/AIOMFAC groups introduced by Marcolli & Peter (2005) are treated separately and are not part of this group (except of alkyl attached to OH);
# 
# The first column entry is the AIOMFAC subgroup ID (or a special ID for certain exception cases), 
# the second column is the SMARTS matching pattern.
#

SMARTS_AIOMFAC = [
[172, '[CH0;X3](=O)OO[NX3;0,+1](=O)-,=[O;0,-1]'],       #Peroxy acyl nitrate C(=O)OONO2 subgroup 
[155, '[C;H2,H3][OH0X2][NX3;+0,+1](=O)-,=[O;0,-1]'],    #Organonitrate CH2ONO2 subgroup (also CH3ONO2 as exception case)
[156, '[CH1][OH0X2][NX3;+0,+1](=O)-,=[O;0,-1]'],        #Organonitrate CHONO2 subgroup
[157, '[CH0][OH0X2][NX3;+0,+1](=O)-,=[O;0,-1]'],        #Organonitrate CONO2 subgroup
[157, '[OH0;X2][OH0X2][NX3;+0,+1](=O)-,=[O;0,-1]'],     #*special peroxide-organonitrate -O-ONO2 subgroup mapped as CONO2 group due to lack of such a subgroup;
[1550,'[OH0X2][NX3;+0,+1](=O)-,=[O;0,-1]'],             #*special organonitrate -ONO2 subgroup without the CH2 group (remove one CHn later if possible);
[54,  '[CH3;X4][NX3;+0,+1](=O)-,=[O;0,-1]'],            #CH3NO2 nitro group
[55,  '[CH2;X4][NX3;+0,+1](=O)-,=[O;0,-1]'],            #CH2NO2 nitro group
[56,  '[C;H1,H0][NX3;+0,+1](=O)-,=[O;0,-1]'],           #CHNO2 nitro group (or as exception: CNO2)
[57,  '[cH0][NX3;+0,+1](=O)-,=[O;0,-1]'],               #aromatic nitro group ACNO2
[560, '[NX3;+0,+1](=O)-,=[O;0,-1]'],                    #pure nitro group; exception for one carbon having two such groups                                                                                                                            
[28,  '[CH3;X4]-[NH2;X3]'],                             #primary amine CH3NH2 subgroup
[29,  '[CH2;X4]-[NH2;X3]'],                             #primary amine CH2NH2 subgroup
[30,  '[C;H1,H0;X4]-[NH2;X3]'],                         #primary amine CHNH2 subgroup; as exception also for CH0-NH2
[31,  '[CH3;X4]-[NH1;X3]'],                             #secondary amine CH3NH subgroup
[32,  '[CH2;X4]-[NH1;X3]'],                             #secondary amine CH2NH subgroup
[33,  '[C;H1,H0;X4]-[NH1;X3]'],                         #secondary amine CHNH subgroup; as exception also for CH0-NH
[1027,'[c]1:[c;H0;X3]:[o]:[c]:[c]1'],                   #furfural variant, mapped as 2 AC + 1 ACH + 1 ether subgroup
[1026,'[c;H0;X3]1:[c]:[o]:[c]:[c]1'],                   #furfural, mapped as 2 AC + 1 ACH + 1 ether subgroup
[926, '[c]1:[c]:[o]:[c]:[c]1'],                         #furan, mapped as 3 ACH subgroups + 1 ether subgroup                                                                                                                       
[161, '[CX3](=[OH0])[OH0;X2][OH1;X2]'],                 #Peroxy Acid C(=O)OOH subgroup
[158, '[C;H2,H3][OH0;X2][OH1]'],                        #Hydroperoxide CH2OOH subgroup; (as exception also CH3OOH)
[159, '[CH1][OH0;X2][OH1]'],                            #Hydroperoxide CHOOH subgroup
[160, '[CH0,cH0][OH0;X2][OH1]'],                        #Hydroperoxide COOH subgroup (or as exception also aromatic hydroperoxide cOOH)
[1580,'[$([OH0;X2]-[C])][OH1]'],                        #*special CHn-ignored hydroperoxide -O-OH subgroup, mapped as -CH2-O-OH subgroup minus 1 alkyl subgroup (CH2 if possible);
[43,  '[CH1;X3](=O)[OH1;X2]'],                          #Formic acid HC(=O)OH subgroup/molecule
[137, '[CH0;X3](=O)[OH1;X2]'],                          #Carboxylic acid C(=O)OH subgroup
[2224,'[CX4][CH0;X3](=O)[OH0;X2]-[OH0;X2;A]'],          #*special perester group CHnC(=O)-O-O- as exception to ester group; 
                                                        #mapped as CH3COO subgroup + CH3O- subgroup minus -CH3 subgroup to account for correct number of C and O atoms;
[2220,'[CX3;H0,H1](=O)[CH0;X3](=O)[OH0;X2]-[OH0;X2;A]'],#*special perester + carbonyl group CHn(=O)C(=O)-O-O- as exception; using aldehyde group and deducting CHn group;

[2022,'[CX4]-[CH0;X3](=O)[CH0;X3](=O)[OH0;X2]'],        #*special ester + aldehyde group for CH0COO + CHn=O combination of subgroups as exceptions;
[2022,'[CH1;X3](=O)[CH0;X3](=O)[OH0;X2]-[C]'],          #*special ester + aldehyde group for CH1COO + CHn=O variant combination of subgroups as exceptions;
[21,  '[CX4;H3][CH0;X3](=O)[OH0;X2]'],                  #ester CH3COO subgroup;
[22,  '[CX4;H2][CH0;X3](=O)[OH0;X2]'],                  #ester CH2COO subgroup;
[22,  '[C;H1,H0][CH0;X3](=O)[OH0;X2]'],                 #*ester CH2COO subgroup as exceptions also mapped for CH1COO and CH0COO;
[18,  '[CH3;X4][CH0;X3](=[OX1])'],                      #ketone CH3C(=O) subgroup ; old: '[C!$([C](O)~[!H!C]);H3][$([C;X3;H0])](=[O;X1])'
[19,  '[C;H2,H1;X4][CH0;X3](=[OX1])'],                  #ketone CH2C(=O) subgroup (also CH1C(=O) as exception case)
[20,  '[CH1;X3;+0,+1](=O)'],                            #aldehyde -CH(=O) subgroup
[20,  '[CH0;X3](=O)'],                                  #special aldehyde subgroup if ketone group cannot be used for a carbonyl in highly multi-functional structure;
[20,  '[CH2]=O'],                                       #formaldehyde; special CH2=O subgroup mapped to subgroup 20;
[162, '[CH3][OH0;X2][OH0;X2][CH3]'],                    #peroxide CH3OOCH3 subgroup
[163, '[CH3][OH0;X2][OH0;X2][CH2]'],                    #peroxide CH3OOCH2 subgroup
[164, '[CH3][OH0;X2][OH0;X2][CH1]'],                    #peroxide CH3OOCH subgroup
[165, '[CH3][OH0;X2][OH0;X2][CH0]'],                    #peroxide CH3OOC subgroup
[166, '[CH2][OH0;X2][OH0;X2][CH2]'],                    #peroxide CH2OOCH2 subgroup
[167, '[CH2][OH0;X2][OH0;X2][CH1]'],                    #peroxide CH2OOCH subgroup
[168, '[CH2][OH0;X2][OH0;X2][CH0]'],                    #peroxide CH2OOC subgroup
[169, '[CH1][OH0;X2][OH0;X2][CH1]'],                    #peroxide CHOOCH subgroup
[170, '[CH1][OH0;X2][OH0;X2][CH0]'],                    #peroxide CHOOC subgroup
[171, '[CH0][OH0;X2][OH0;X2][CH0]'],                    #peroxide COOC subgroup
[1700,'[C;H0,H1,H2][OH0;X2][OH0;X2]'],                  #*special peroxide CHn-O-O- subgroup when second carbon atom at end is already matched to 
                                                        #another group; in this case we map CHnOO group to the CHOOC subgroup and 
                                                        #deduct one alkyl (CHn) group detected later (if possible);
[1710,'[OH0;X2;A]-[OH0;X2;A]'],                         #*special peroxide -O-O- subgroup when both carbon atoms at ends are already matched to 
                                                        #other groups; in this case we map -O-O- group to the COOC subgroup and 
                                                        #deduct two alkyl (CHn) groups detected later (if possible);
[1120, '[CX4;H2]([OX2;H1])[OX2;H1]'],                   #geminal diol case (CH2 aliphatic)
[1121, '[CX4;H1]([OX2;H1])[OX2;H1]'],                   #geminal diol case (CH1 aliphatic)
[1122, '[CX4;H0]([OX2;H1])[OX2;H1]'],                   #geminal diol case (C aliphatic)
[154, '[CH2;X4;R0][OH0;X2;R0][CH2;X4;R0;$([CH2][OH0][CH2][CH2][OH0][CH2])]'],  #oxyethylene group (-CH2-O-CH2-) in oligomers like PEG 
                                                        #(has to have at least two of these groups in succession);
[27,  '[CX4;H2;r5;R1;$([CH2;R1][CH2;R1][CH2;R1])][OH0;r5;R1]'], #tetrahydrofuran (oxolane), special ether group: THF[CH2O]
[24,  '[CX4;H3][OH0;X2]'],                              #ether CH3O-
[25,  '[$([CX4;H2]([A!O]))][OH0;X2]'],                  #ether -CH2O- (first prefer a matching option where the C is not bonded to an OH group)
[25,  '[CX4;H2][OH0;X2]'],                              #ether -CH2O-
[26,  '[$([CX4;H1]([A!O])[A!O])][OH0;X2]'],             #ether >CHO-, (first prefer a matching option where the C is not bonded to an OH group);
[26,  '[CX4;H1,H0][OH0;X2]'],                           #ether >CHO-, as exception also for ether C-O- (with zero H);
[5,   '[CH2]=[CH1]'],                                   #Alkene CH2=CH subgroup
[6,   '[CH1]=[CH1]'],                                   #Alkene CH=CH subgroup
[7,   '[CH2]=[CH0]'],                                   #Alkene CH2=C subgroup
[8,   '[CH1]=[CH0]'],                                   #Alkene CH=C subgroup
[70,  '[CH0]=[CH0]'],                                   #Alkene C=C subgroup
[149, '[$([CX4;H3]([OH1X2]))]'],                        #CH3 alkyl attached to OH (while not counting OH)
[150, '[$([CX4;H2]([OH1X2]))]'],                        #CH2 alkyl attached to OH (while not counting OH)
[151, '[$([CX4;H1]([OH1X2]))]'],                        #CH1 alkyl attached to OH (while not counting OH)
[152, '[$([CX4;H0]([OH1X2]))]'],                        #>C< alkyl attached to OH (while not counting OH)
[17,  '[cX3][OH1;X2]'],                                 #aromatic carbon alcohol ACH subgroup (phenol) 
[153, '[OH1;X2;A]'],                                    #-OH group; must always be attached to aliphatic carbon.
[9,   '[cH1]'],                                         #Aromatic Hydrocarbon ACH subgroup
[10,  '[cH0]'],                                         #Aromatic Hydrocarbon AC subgroup
[1,   '[CX4;H3,H4]'],                                   #CH3 standard alkyl; as exception also for CH4
[2,   '[CH2]'],                                         #CH2 standard alkyl
[3,   '[CH1]'],                                         #CH1 standard alkyl
[4,   '[CH0]'],                                         #C standard alkyl
[2502,'[OH0;X2]'],                                      #exception case; map ether oxygen without carbon as ether -CH2O- minus CHn
[2501,'[oH0;X2]'],                                      #exception case; map aromatic oxygen without carbon as ether -CH2O- minus CHn
[2002,'[OX1;H0]']                                       #exception case; map carbonyl '=O' as aldehyde group minus the CHn group;
]    #closing the 2D list