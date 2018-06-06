import numpy as np
from spipy.merge import utils as merge
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
	
	fpath = '../phase/3dvolume.bin'
	# quats : 5 columns, the first four are w, qx, qy, qz, the last one is orientation sampling weight
	quats = np.loadtxt('quaternions_6.txt')
	# make mask
	mask = np.zeros((81,81))
	mask[39:42,:] = 1
	
	print("\n-- Generate 2000 slices from 3D model ..")
	model_0 = np.fromfile(fpath).reshape((125,125,125))
	t1 = time.time()
	slices = merge.get_slice(model=model_0, quaternions=quats[:2000,:4], det_size=[81,81], det_center=None, mask=mask)
	print("Done. Time : "+str(time.time()-t1)+" s")
	plt.subplot(1,2,1)
	plt.imshow(np.log(1+slices[0]))
	plt.title('pattern index=0')
	plt.subplot(1,2,2)
	plt.imshow(np.log(1+slices[1999]))
	plt.title('pattern index=1999')
	plt.show()

	print("\n-- Merge the generated 2000 patterns into a new model ..")
	model_1 = np.zeros(model_0.shape)
	t1 = time.time()
	merge.merge_slice(model=model_1, quaternions=quats[:2000,:4], slices=slices, weights=None, det_center=None, mask=mask)
	print("Done. Time : "+str(time.time()-t1)+" s")
	plt.figure(figsize=(10,3))
	plt.subplot(1,3,1)
	plt.imshow(np.log(1+model_1[62,:,:]))
	plt.title('Y-Z Plain of Merged Model')
	plt.subplot(1,3,2)
	plt.imshow(np.log(1+model_1[:,62,:]))
	plt.title('X-Z Plain of Merged Model')
	plt.subplot(1,3,3)
	plt.imshow(np.log(1+model_1[:,:,62]))
	plt.title('X-Y Plain of Merged Model')
	plt.show()
	
	print("\n-- Calculate poisson likelihood between slices and pattern ..")
	from spipy.image.preprocess import adu2photon
	pat = merge.get_slice(model=model_0, quaternions=quats[3000,:4], det_size=[81,81], det_center=None, mask=mask)
	adu, pat = adu2photon(dataset=np.array([pat]), mask=mask, photon_percent=0.1, nproc=1, transfer=True, force_poisson=True)
	R_jk = np.zeros(len(slices))
	for ind, s in enumerate(slices):
		R_jk[ind] = merge.poisson_likelihood(W_j=s, K_k=pat[0], beta=5, weight=quats[ind,4])
	P_jk = R_jk/np.sum(R_jk)
	plt.hist(P_jk, bins=30)
	plt.title("Normalized poisson-likelihood probabilities")
	plt.show()