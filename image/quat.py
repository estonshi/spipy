import numpy as np

def help(module):
	if module=="invq":
		print("This function is used to calculate the inverse/reciprocal of a quaternion")
		print("    -> Input: q (np.ndarray([w,qx,qy,qz]))")
		print("[Notice] w = cos(the/2) , q? = ? * sin(theta/2). BE CAREFUL of the order!")
		return
	elif module=="quat_mul":
		print("This function is used to multiply two quaternions")
		print("    -> Input: q1 (np.ndarray([w,qx,qy,qz]))")
		print("              q2 (np.ndarray([w,qx,qy,qz]))")
		print("[Notice] w = cos(the/2) , q? = ? * sin(theta/2). BE CAREFUL of the order!")
		return
	elif module=="conj":
		print("This function is used to calculate the conjugate quaternion")
		print("    -> Input: q (np.ndarray([w,qx,qy,qz]))")
		print("[Notice] w = cos(the/2) , q? = ? * sin(theta/2). BE CAREFUL of the order!")
		return
	elif module=="quat2azi":
		print("Transfer a quaternion to azimuth angle")
		print("    -> Input: q (np.ndarray([w,qx,qy,qz]))")
		print("    -> Output: azi (np.array([theta,x,y,z]))")
	elif module=="azi2quat":
		print("Transfer a azimuth angle to quaternion")
		print("    -> Input: azi (np.array([theta,x,y,z]))")
		print("    -> Output: q (np.ndarray([w,qx,qy,qz]))")
	elif module=="quat2rot":
		print("This function is used to transfer a quaternion to a 3D rotation matrix")
		print("    -> Input: q (np.ndarray([w,qx,qy,qz]))")
		print("    -> Return dtype: numpy.matrix(3x3)")
		print("[Notice] w = cos(the/2) , q? = ? * sin(theta/2). BE CAREFUL of the order!")
		return
	elif module=="rotv":
		print("This function is used to rotate a 3D vector using a quaternion")
		print("    -> Input: vector (np.ndarray([x,y,z]))")
		print("              q (quaternion : np.array([w,qx,qy,qz]))")
		print("    -> Return dtype: np.ndarray([x',y',z'])")
		print("[Notice] w = cos(the/2) , q? = ? * sin(theta/2). BE CAREFUL of the order!")
		return
	elif module=="Slerp":
		print("This function is used to calculate linear spherical interpolation between two quaternions")
		print("    -> Input: q1 (np.ndarray([w,qx,qy,qz]))")
		print("              q2 (np.ndarray([w,qx,qy,qz]))")
		print("              t  (interpolation weight from q1 to q2, 0~1)")
		print("[Notice] w = cos(the/2) , q? = ? * sin(theta/2). BE CAREFUL of the order!")
		return
	else:
		raise ValueError("No module names "+str(module))

# quaternion inverse
def invq(q):
	q = q.astype(float)
	conjX = conj(q)
	mod = np.linalg.norm(q)
	return conjX/mod

# quaternion multiply X*Y
def quat_mul(q1, q2):
	q0 = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	q1 = q1[0]*q2[1]+q1[1]*q2[0]+q1[2]*q2[3]-q1[3]*q2[2]
	q2 = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	q3 = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return np.array([q0,q1,q2,q3])

# conjugate quaternion 
def conj(q):
	return np.array([q[0],-q[1],-q[2],-q[3]])

# transfer between azimuth and quat
def quat2azi(q):
	theta = np.arccos(q[0])*2
	n = np.array(q[1:])/np.sin(theta/2)
	return np.array([theta,n[0],n[1],n[2]])

def azi2quat(azi):
	w = np.cos(azi[0]/2)
	x = np.sin(azi[0]/2) * azi[1:]
	return np.array([w,x[0],x[1],x[2]])
# transfer between quat and rotation matrix
def quat2rot(q):
	rot_m = np.matrix([[1-2*q[2]**2-2*q[3]**2, 2*q[1]*q[2]-2*q[0]*q[3], 2*q[1]*q[3]+2*q[0]*q[2]],
			[2*q[1]*q[2]+2*q[0]*q[3], 1-2*q[1]**2-2*q[3]**2, 2*q[2]*q[3]-2*q[0]*q[1]], 
			[2*q[1]*q[3]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], 1-2*q[1]**2-2*q[2]**2]])
	return rot_m

def rot2quat(rot):
	w = 0.5*np.sqrt(1+rot[0,0]+rot[1,1]+rot[2,2])
	qx = (rot[2,1]-rot[1,2])/(4*w)
	qy = (rot[0,2]-rot[2,0])/(4*w)
	qz = (rot[1,0]-rot[0,1])/(4*w)
	return np.array([w,qx,qy,qz])

# rotate a vector
def rotv(vector, q):
	newvec = quat2rot(q) * vector.reshape((3,1))
	return np.array(newvec).T[0]

# Spherical interpolation between q1 and q2
def Slerp(q1, q2, t):
	theta = np.arccos(np.dot(q1,q2))
	qt = np.sin((1-t)*theta)/np.sin(theta) * q1 + np.sin(t*theta)/np.sin(theta) * q2
	return qt
