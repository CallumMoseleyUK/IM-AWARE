import pandas as pd
from pathlib import Path 
import re

#functions to import data from the ANM into pandas databases, The only processing to be done is removing any 
#lines before the headings in excel, namely the logo of the ANM and the download date. 

def coords(rawCords):
    Astr = rawCords.split('°')
    deg = int(Astr[0])
    Bstr = Astr[1].split("'")
    mn = int(Bstr[0])
    Cstr = Bstr[1].split('"')
    scn = float(Cstr[0])
    if deg >= 0 :
        fcord = float( scn /3600 + mn/60 + deg)
    if deg < 0 :
        fcord =  float( deg -  mn/60 - scn /3600 )
    return fcord
    

def cleanTailingsVol(vol):
    vol = vol.split(',')[0].split('.')
    powers3 = [3*exp for exp in list(range(0,len(vol)))] 
    powers3 =  list(reversed(powers3))
    endVol = 0
    for i in range(len(vol)):
        endVol += 10**(powers3[i])*int(vol[i])
    return endVol 


def coding(file):
    #this function returns the dictionaries from the coding information provided in the file
    df = pd.read_excel(file)
    oCode = list(df['Old_Code'].values)
    nCode = list(df['New_Code'].values)
    out = {}
    for i in range(len(oCode)):
        out[ oCode[i] ] = nCode[i]
    return out

def old_codes(rawdata,field,codingF):
    #this identified unique values of categorical variables of a given field, considering raw data provided.
    #the output will be saved in the codingF folder 
    codes = rawdata[field].drop_duplicates()
    codes.to_excel(Path.joinpath(codingF,f'{field}.xlsx') , index=False) 

def clean_ID_field(arg):
    return re.sub(r"[^a-zA-Z0-9]","", str(arg.encode('utf-8')))

def genDamID(keys):
    return '{}_{}'.format(clean_ID_field(keys[0])[1:8], clean_ID_field(keys[1])[1:8])

def check_id_unique(idList):
    flag=0
    for i in range(len(idList)):
        for j in range(len(idList)):
            if i != j:
                if idList[i] == idList[j]:
                    try:
                        subStr = '{}_{}'.format(idList[j].split('_')[0],idList[j].split('_')[1])
                        ind = int(idList[j].split('_')[2])
                    except:
                        subStr = idList[i]
                        ind = 1
                    idList[j] = '{}_{}'.format(subStr,ind+1)
    return idList

def clearANMData(rawFile):

    colNames =['Dam_Name','Company', 'CPF_CNPJ','Latitude','Longitude','Tailings_Material','Height',
    'Stored_Volume','Building_Method','Risk','Potential_Damage','Type','PNSB','Emergency_Level','Current_Status']

    code1 = {'Minério de Estanho Primário': 'tin', 'Minério de Ferro': 'iron', 'Areia': 'sand', 'Argila': 'clay', 
    'Saibro': 'clay', 'Aluvião com Gemas': 'gem_alluvium', 'Calcário': 'limestone', 'Minério de Manganês': 'Mn', 
    'Rocha Aurífera': 'gold_rock', 'Bauxita Grau Metalúrgico': 'bauxite', 'Xisto': 'shale', 'Minério de Ouro Primário': 'gold', 
    'Minério com Gemas': 'gen_rock', 'Caulim': 'kaolin', 'Carvão Mineral Camada Bonito': 'coal_clay', 
    'Carvão Mineral Camada Barro Branco': 'coal', 'Carvão Mineral': 'coal', 'Dolomito': 'dolomite', 'Calcário Dolomítico': 'dolomite', 
    'Agalmatolito': 'agalmatolite', 'Fluorita': 'fluorite', 'Magnesita': 'magnesite', 'Minério de Cromo': 'chrome', 
    'Minério de Cobre': 'copper', 'Argila Arenosa': 'sandy_clay', 'Aluvião Diamantífero': 'gem_alluvium', 
    'Argila Refratária': 'clay', 'Areia e Cascalho': 'gravely_sand', 'Aluvião Aurífero': 'gold_alluvium', 
    'Minério de Nióbio': 'Niobium', 'Rocha Fosfática': 'phosphate_rock', 'Minério de Zinco': 'Zn', 'Hematita': 'hematite', 'Fosfato': 'phospate', 
    'Rocha Carbonática': 'coal', 'Arenito': 'sandstone', 'Minério de Ouro Secundário': 'gold', 'Itabirito': 'itarbirite', 
    'Bauxita Grau Não Metalúrgico': 'bauxite', 'Minério de Vanádio': 'V', 'Granito': 'granite', 
    'Cascalho': 'gravel', 'Areia com Minerais Pesados': 'heavy_sand', 'Vermiculita': 'vermiculite', 
    'Aluvião Estanífero': 'tin_alluvium', 'Minério de Níquel': 'Ni', 'Bentonita': 'bentonite', 'Cromita': 'chromite', 
    'Minério de Estanho Secundário': 'tin_rock', 'Quartzito': 'quartz', 'Filito': 'philite', 'Serpentinito': 'serpentinite', 'Silvanita': 'silvanite', 
    'Sais': 'salt', 'Areia Quartzosa': 'quartz_sand', 'Argila Caulinítica': 'clay', 'Areia Industrial': 'ind_sand', 
    'Rocha Diamantífera': 'gem_rock', 'Granulito Gnaisse': 'gness', 'Pegmatito': 'pegmatite'}

    code2 = { 'Indefinido': 'unknown' ,'0 - Etapa única': 'single_stage', '5 - Alteamento por linha de centro' : 'centerline', 
    '2 - Alteamento a jusante': 'downstream', '10 - Alteamento a montante ou desconhecido' : 'upstream' } 

    code3 = {'Baixo':'low', 'Médio':'medium', 'Alto':'high'}
    code4 = {'Médio':'medium','Alto':'high','Baixo':'low'}
    code5 = {'Não':False, 'Sim':True}
    code6 = {'Sem emergência':0, 'Nível 1': 1, 'Nível 3':3, 'Nível 2':2}
    code7 = {'1º Campanha 2021 - Atestado': 'certified_2021', '1º Campanha 2021 - Não Atestado' : 'noncertified_2021', 
    'Revisão Periódica - Atestado': 'certified_periodically', 'Extraordinária ou Exigência de Fiscalização - Atestado' : 'certified_emergency', 
    '1º Campanha 2021 - Não Enviado': 'noncertified_2021', 'Revisão Periódica - Não Atestado' :'noncertified_periodically', 
    'Extraordinária ou Exigência de Fiscalização - Não Atestado' : 'noncertified_periodically' }

    df = pd.read_excel(rawFile,engine='openpyxl')
    cols = df.columns
    delCols = [cols[5],cols[6],cols[7]]
    df = df.drop(columns = delCols )
    df.columns = colNames
    df['Stored_Volume'] = df['Stored_Volume'].astype(str).apply(cleanTailingsVol)
    df['Lat'] = df['Latitude'].astype(str).apply(coords)
    df['Long'] = df['Longitude'].astype(str).apply(coords)
    df = df.replace({'Tailings_Material': code1}) 
    df = df.replace({'Building_Method': code2})
    df = df.replace({'Risk': code3})
    df = df.replace({'Potential_Damage': code4})
    df = df.replace({'PNSB': code5})
    df = df.replace({'Emergency_Level': code6})
    df = df.replace({'Current_Status': code7})
    cStatus = list(df['Current_Status'])
    status = []
    control = []
    for record in cStatus:
        record = record.split('_')
        if len (record) == 2:
            status.append(record[0])
            control.append(record[1])
        else:
            status.append(pd.NA)
            control.append(pd.NA)

    df['Current_Status'] = pd.Series(status)
    df['Control'] = pd.Series(control)
    df = df.drop(columns = ['Latitude','Longitude'])
    df['ID'] = df[['Dam_Name','Company']].apply(genDamID ,axis=1)
    idList = df['ID'].tolist()
    idList = check_id_unique(idList)
    df['ID'] = idList
    outFname = str(rawFile.name).split('.')[0]
    directory = Path(__file__).parent
    
    outFilePath = directory.joinpath(f'processed_{outFname}.xlsx')
    df.to_excel(outFilePath, index = False )


BaseF = Path(__file__).parent
fName = BaseF.joinpath('ANM 06-2021.xlsx')
clearANMData(fName)







