import matplotlib.pyplot as plt
import DBdriver.DBFunctions as dbf
from dam_break.dambreak_lib import DAMBREAK_STAT

table = 'Flooding_Model_Description'

damList = dbf.get_all_dams(table)

figEnergy,axEnergy = plt.subplots()
figSpeed,axSpeed = plt.subplots()
figArea,axArea = plt.subplots()

for damID in damList:
    setRecordList = dbf.query_by_type('Monte_Carlo',damID=damID)
    setRecord = setRecordList[1] #hard-coded to use the 200 simulation sets

    damBreakStat = DAMBREAK_STAT(setRecord)

    damping = damBreakStat.get_data_set('Damping')
    pEnergy,xEnergy = damBreakStat.calculate_ecdf('Total_Energy')
    pSpeed,xSpeed = damBreakStat.calculate_ecdf('Max_Velocity')
    pArea,xArea = damBreakStat.calculate_ecdf('Flooding_Area')

    axEnergy.plot(xEnergy,pEnergy,label=damID)
    axSpeed.plot(xSpeed,pSpeed,label=damID)
    axArea.plot(xArea,pArea,label=damID)
    
axEnergy.legend(damList,bbox_to_anchor=(1.1, 1.05))
axSpeed.legend(damList,bbox_to_anchor=(1.1, 1.05))
axArea.legend(damList,bbox_to_anchor=(1.1, 1.05))

axEnergy.set_xlabel('Energy (J)')
axSpeed.set_xlabel('Speed (m/s)')
axArea.set_xlabel('Area (m^2)')
axEnergy.set_ylabel('P(x)')
axSpeed.set_ylabel('P(x)')
axArea.set_ylabel('P(x)')

    