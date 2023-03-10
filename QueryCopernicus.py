import requests
import xmltodict
import json

def query (loc,dates,cloudcover): 
    q = ''
    q += f'footprint:"intersects({loc[0]},{loc[1]})" AND ' 
    q += f'ingestiondate:[{dates[0]}T00:00:00.000Z TO {dates[1]}T00:00:00.000Z] AND '
    q += f'producttype:S2MSI2A AND '  
    q += f'cloudcoverpercentage:[0 TO {cloudcover}]'
    #q += f' AND filename:*10m.jp2'
    print(q)
    return q

def urlQry (loc,dates,cloudcover):
   url = 'https://scihub.copernicus.eu/dhus/search?start=0&rows=100&q='
   q = query(loc,dates,cloudcover) 
   return url + q 

def figurelist(loc,dates,cloudCover):
    r=requests.get(url = urlQry(loc,dates,cloudCover) , auth = ('andalon','Bdate81V05'))
    data = xmltodict.parse(r.content)
    figs = []
    for figure in data['feed']['entry']:
        figs.append({'title': figure['title'], 'id': figure['id']})
    return figs

brum = [-20.119722,-44.121389]
dates = ['2021-01-01','2021-07-31']
cCover = 10
imList = figurelist(brum,dates,cCover)

print(len(imList))
print (imList[2])

def downMetadata(id,name): 
    urld = f'https://scihub.copernicus.eu/dhus/odata/v1/Products(\'{id}\')/Nodes(\'{name}.SAFE\')/Nodes(\'manifest.safe\')/$value'
    #urld = f'https://scihub.copernicus.eu/dhus/odata/v1/Products(\'{id}\')/$value'
    r=requests.get(url = urld , auth = ('andalon','Bdate81V05'))
    data = xmltodict.parse(r.content)
    return data

outJson = downMetadata(imList[1]['id'],imList[1]['title'])

with open('outJson1.txt','w') as f:
    f.write(json.dumps(outJson,indent=4))

