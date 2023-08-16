import numpy as np
from astropy.table import Table
from astropy.io import ascii as asc
lines_data = asc.read("lines_vac.dat", names=["wavelength", "spectrum"])
lines_data["air"] = np.round(lines_data["wavelength"] / 1.0003, 2)
print(lines_data["air", "spectrum"])

