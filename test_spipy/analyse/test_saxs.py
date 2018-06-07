import numpy as np
from spipy.analyse import saxs
import matplotlib.pyplot as plt
import h5py

if __name__=="__main__":

	dataset = h5py.File('../test_pattern.h5','r')['patterns'][...]
	mask = h5py.File('../test_pattern.h5','r')['mask'][...]

	print("\ncalculate saxs pattern of a dataset ..")
	saxs_pat = saxs.cal_saxs(data=dataset)

	print("\nSearch zero frequency point of saxs pattern ..")
	cen = saxs.frediel_search(pattern=saxs_pat, estimated_center=(64,64), mask=mask)
	print("Center is : " + str(cen))

	print("\nCalculate averaged intensity radial profile of a dataset ..")
	inten_prof = saxs.inten_profile_vfast(dataset, mask, 581, 7.9, 64, 0.6)

	plt.plot(inten_prof[:,0], np.log(1+np.abs(inten_prof[:,1])), 'r-')
	plt.xlabel('q')
	plt.ylabel('radial intensity profile')
	plt.show()

	print("\nEvaluate particle size using saxs profile...")
	[estimated, radial_intensity] = saxs.particle_size(saxs=saxs_pat, estimated_center=cen, exparam='578,7.5,0.6',\
	 high_filter_cut=0.3, power=0.8, mask=mask)
	print("estimated particle size : "+str(estimated)+" nm")
	plt.plot(radial_intensity[:,0], radial_intensity[:,1], 'r-')
	plt.plot(np.zeros(50)+80, np.linspace(0,0.2,50),'k-')
	plt.show()