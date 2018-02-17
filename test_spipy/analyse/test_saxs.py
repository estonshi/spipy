import numpy as np
from spipy.analyse import saxs
import matplotlib.pyplot as plt
import h5py

if __name__=="__main__":

	dataset = h5py.File('../test_data.h5','r')['patterns'][...]

	print("\ncalculate saxs pattern of a dataset ..")
	saxs_pat = saxs.cal_saxs(data=dataset)

	print("\nSearch zero frequency point of saxs pattern ..")
	cen = saxs.frediel_search(pattern=saxs_pat, estimated_center=(130,128))
	print("Center is : " + str(cen))

	print("\nCalculate averaged intensity radial profile of a dataset ..")
	inten_prof = saxs.inten_profile_vfast(dataset, 581, 7.9, 128, 0.3)

	plt.plot(inten_prof[:,0], np.log(np.abs(inten_prof[:,1])), 'r-')
	plt.xlabel('q')
	plt.ylabel('radial intensity profile')
	plt.show()
