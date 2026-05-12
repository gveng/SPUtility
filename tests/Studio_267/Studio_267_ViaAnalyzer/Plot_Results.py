## FILE 1

from datetime import datetime

import emerge_iron
import emerge as em
from emerge.plot import plot_sp

ProjectName = "Interposer_GV_1"
ResultTimeCode = datetime.now().strftime("%Y%m%d-%H%M%S")



m = em.Simulation(ProjectName, load_file=True)
# setup simulation
data = m.data.mw

plot_sp(data.scalar.grid.freq, [data.scalar.grid.S(1,1), data.scalar.grid.S(2,1)], labels=['S11', 'S21'])

data.scalar.grid.export_touchstone(f"{ProjectName}_{ResultTimeCode}.s2p", Z0ref=50, format="RI", custom_comments=["Rafale Interposer","GabrieleV"], funit="HZ" )