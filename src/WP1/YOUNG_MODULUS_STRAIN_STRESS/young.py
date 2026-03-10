#!/usr/bin/python3

from __future__ import print_function
import numpy as np
from numpy import arange
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import glob


#This programme starts plotting the stress-strain diagram, then it calculates the area through an integration using the  trapezoidal rule
#The two arguments of trapezoidal rule are x and y which are the stress and strain, obtained from the read_statis() function. It then integrates.
#This function is called after that the stress-strain curves have been plotted in the code.

def read_statis(direction, final_step, tstep = 50):
	if direction == 'all':
		#print('Per ora facciamo una direzione alla volta')
		filelist = glob.glob('STATIS')
		dati = []
		for filename in filelist:
			try:
				IN = open(filename, newline="")
			except:
				print('There is not the ', filename, ' file')
				exit()
			direction = filename.split('/')[0]
			if final_step == 'all':
				fields = IN.readlines()
				nstep = len(fields)
			else:
				nstep = int(final_step / tstep)
				fields = IN.readlines()[:nstep]
			for line in fields:
				dati.append([float(s) for s in line.split()])
			IN.close()
		mat_in = np.array(dati)
#		print(mat_in.shape)
		mat = np.reshape(mat_in, (3, nstep, 8))
#		print(mat.shape)
	else:
		filename = direction + 'STATIS'
		try:
			IN = open(filename, newline="")
		except:
			print('There is not the STATIS file')
			exit()
		if final_step == 'all':
			fields = IN.readlines()
			nstep = len(fields)
		else:
			nstep = int(final_step / tstep)
			fields = IN.readlines()[:nstep]
		dati=[]
		for line in fields:
			dati.append([float(s) for s in line.split()])
		IN.close()
		mat= np.array(dati)
	return mat

def read_lammps(direction, final_step, tstep = 100):
	if direction == 'all':
		filelist = ['output_young_x', 'output_young_y', 'output_young_z']
		dati = []
		for filename in filelist:
			try:
				IN = open(filename, newline="")
			except:
				print('There is not the ', filename, ' file')
				exit()
			direction = filename.split('.')[0].split('_')[1]
			fields = IN.readlines()
			nlin = len(fields)
			for n in range(nlin):
				if 'Step ' in fields[n]:
					linit = n+1
					labels = fields[n].split()
					for p in range(len(labels)):
						if 'Step' in labels[p]:
							c_step = p
						elif 'TotEng' in labels[p]:
							c_te = p
						elif 'Pxx' in labels[p]:
							c_pxx = p
						elif 'Pyy' in labels[p]:
							c_pyy = p
						elif 'Pzz' in labels[p]:
							c_pzz = p
						elif 'Lx' in labels[p]:
							c_lx = p
						elif 'Ly' in labels[p]:
							c_ly = p
						elif 'Lz' in labels[p]:
							c_lz = p
				elif 'Loop time' in fields[n]:
					lend = n-1
			if final_step == 'all':
				nstep = lend - linit
			else:
				nstep = int(final_step / tstep)	
			for i in range(linit,linit+nstep):
				line = fields[i].split()
				for p in [c_step, c_lx, c_ly, c_lz, c_pxx, c_pyy, c_pzz, c_te]:
					dati.append(float(line[p]))
				#dati.append(line[0], line[-1], line[-2], line[-3], line[5], line[6], line[7], line[1])
				
			IN.close()
		mat_in = np.array(dati)
#		print(mat_in.shape)
		mat = np.reshape(mat_in, (3, nstep, 8))
#		print(mat.shape)
	else:
		filename = 'out_'+direction+'.lammps'
		try:
			IN = open(filename, newline="")
		except:
			print('There is not the STATIS file')
			exit()
		fields = IN.readlines()
		nlin = len(fields)
		for n in range(nlin):
			if 'Step ' in fields[n]:
				linit = n+1
				labels = fields[n].split()
				for p in range(len(labels)):
					if 'Step' in labels[p]:
						c_step = p
					elif 'TotEng' in labels[p]:
						c_te = p
					elif 'Pxx' in labels[p]:
						c_pxx = p
					elif 'Pyy' in labels[p]:
						c_pyy = p
					elif 'Pzz' in labels[p]:
						c_pzz = p
					elif 'Lx' in labels[p]:
						c_lx = p
					elif 'Ly' in labels[p]:
						c_ly = p
					elif 'Lz' in labels[p]:
						c_lz = p
			elif 'Loop time' in fields[n]:
				lend = n-1
		if final_step == 'all':
				nstep = lend - linit
		else:
			nstep = int(final_step / tstep)	
		for i in range(linit,linit+nstep):
			line = fields[i].split()
			for p in [c_step, c_lx, c_ly, c_lz, c_pxx, c_pyy, c_pzz, c_te]:
				dati.append(float(line[p]))

		IN.close()
		mat = np.array(dati)
	return mat

def udm(dat, unit='KAtm'):
	if unit == 'KAtm':
		conv = 0.101325
	# Transform to KAtm to GPa
	elif unit == 'KPa':
		conv = 0.0001
	else:
		print('Unità di misura non supportata,')
		print('inserire fattore di conversione')
		conv = input('per ottenere GPa\n')
		conv = float(conv)
	arr = dat*(-1)*conv
		
	return arr
def objective(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d
    
#def integrate_curve(x, y):
#    """Integrate stress-strain curve using trapezoidal rule"""
#    return np.trapz(y, x)
    
    
while True:
	direction = input("Enter the direction (x, y, z, or all):\n ")
	if direction == 'all':
		outfile_x = 'Stress_strain_x.dat'
		outfile_y = 'Stress_strain_y.dat'
		outfile_z = 'Stress_strain_z.dat'
		OUTx = open(outfile_x, 'w')
		OUTy = open(outfile_y, 'w')
		OUTz = open(outfile_z, 'w')
		while True:
			final_step = input("Enter the final step:\n ")
			if final_step == 'all':
				swtype = input("DLPOLY (1) or LAMMPS (2)?\n")
				if swtype == "DLPOLY" or swtype == "1":
					mall = read_statis(direction,final_step)
				elif swtype == "LAMMPS" or swtype == "2":
					mall = read_lammps(direction,final_step)
				break
			elif final_step == '':
				print ("Insert an integer or type 'all' to consider the full trajectory")
			else:
				final_step = int(final_step)
				swtype = input("DLPOLY (1) or LAMMPS (2)?\n")
				if swtype == "DLPOLY" or swtype == "1":
					mall = read_statis(direction,final_step)
				elif swtype == "LAMMPS" or swtype == "2":
					mall = read_lammps(direction,final_step)
				break
		X, Y, Z = mall[0], mall[1], mall[2]
		xx, xy, xz, sx = X[:,1], X[:,2], X[:,3], udm(X[:,4], unit='KPa')
		Ax = xy * xz
		Sx = sx * (Ax / Ax[0])
		xx = (xx - xx[0]) / xx[0]
		yx, yy, yz, sy = Y[:,1], Y[:,2], Y[:,3], udm(Y[:,5], unit='KPa')
		Ay = yx * yz
		Sy = sy * (Ay / Ay[0])
		yy = (yy - yy[0]) / yy[0]
		zx, zy, zz, sz = Z[:,1], Z[:,2], Z[:,3], udm(Z[:,6], unit='KPa')
		Az = zx * zy
		Sz = sz * (Az / Az[0])
		zz = (zz - zz[0]) / zz[0]
		print('#Strain   Stress', file = OUTx)
		print('#Strain   Stress', file = OUTy)
		print('#Strain   Stress', file = OUTz)
		for i in range(len(xx)):
			print(xx[i], Sx[i], file = OUTx)
			print(yy[i], Sy[i], file = OUTy)
			print(zz[i], Sz[i], file = OUTz)
		OUTx.close
		OUTy.close
		OUTz.close
		
#		# area under stress-strain curve
#		Ax = np.trapz(Sx, xx)
#		Ay = np.trapz(Sy, yy)
#		Az = np.trapz(Sz, zz)
#		print("Area under stress-strain curve for x direction: {Ax}") 
#		print("Area under stress-strain curve for y direction: {Ay}")
#		print("Area under stress-strain curve for z direction: {Az}")
		
	        # curve fit
		poptx, _ = curve_fit(objective, xx, Sx, p0= [1000,-100,100,0], maxfev=5000)
		popty, _ = curve_fit(objective, yy, Sy, p0= [1000,-100,100,0], maxfev=5000)
		poptz, _ = curve_fit(objective, zz, Sz, p0= [1000,-100,100,0], maxfev=5000)
		# summarize the parameter values and calculate the error
		# print a summary and a data file
		ax, bx, cx, dx = poptx
		az, bz, cz, dz = poptz
		ay, by, cy, dy = popty
		young = (cx + cy + cz)/3
		stdev = (((cx - young)**2 + (cy - young)**2 + (cz - young)**2)/3)**0.5
		
		fig, (ax1, ax2, ax3) = plt.subplots(1,3)
		fig.suptitle('Stress Strain plots in the three directions')
		ax1.plot(xx, Sx, c='b', label = 'MD')
		Sxf = objective(xx, ax, bx, cx, dx)
		ax1.plot(xx, Sxf, c='r', label = 'Fit')
		ax1.set_title('X')
		#ax1.set_xlabel('Strain on X')
		ax1.set_ylabel('Stress')
		ax1.legend()
		ax2.plot(yy, Sy, c='b', label = 'MD')
		Syf = objective(yy, ay, by, cy, dy)
		ax2.plot(yy, Syf, c='r', label = 'Fit')
		ax2.set_title('Y')
		ax2.set_xlabel('Strain')
		#ax2.set_ylabel('Stress on Y')
		ax3.plot(zz, Sz, c='b', label = 'MD')
		Szf = objective(zz, az, bz, cz, dz)
		ax3.plot(zz, Szf, c='r', label = 'Fit')
		ax3.set_title('Z')
		#ax3.set_xlabel('Strain on Z')
		#ax3.set_ylabel('Stress on Z')
		figure_name = 'Stress_Strain_3_dimensions_' + str(final_step)
		plt.savefig(figure_name, dpi=300)
		
		print("Fitting functions:")
		print('Stress X = %.10E x^3 + %.10E x^2 + %.10E x + %.10E' % (ax, bx, cx, dx))
		print('Stress Y = %.10E x^3 + %.10E x^2 + %.10E x + %.10E' % (ay, by, cy, dy))
		print('Stress Z = %.10E x^3 + %.10E x^2 + %.10E x + %.10E' % (az, bz, cz, dz))
		print('**************************************')
		print('Young modulus in X direction is: ', cx)
		print('Young modulus in Y direction is: ', cy)
		print('Young modulus in Z direction is: ', cz)
		print('**************************************')
		print('Average Young modulus:\n ', young)
		print('Standard deviation:\n ', stdev)
		
		plt.show()
		break
	elif direction == 'x' or direction == 'y' or direction == 'z':
		outfile = 'Stress_strain_' + direction + '.dat'
		OUT = open(outfile, 'w')
		while True:
			final_step = input("Enter the final step:\n ")
			if final_step == 'all':
				mat = read_statis(direction, final_step)
				break
			elif final_step == '':
				print ("Insert an integer or type 'all' to consider the full trajectory")
			else:
				final_step = int(final_step)
				mat = read_statis(direction, final_step)
				break

		if direction == 'x':
			elon, dir2, dir3, s = mat[:,1], mat[:,2], mat[:,3], udm(mat[:,4])
		elif direction == 'y':
			elon, dir2, dir3, s = mat[:,2], mat[:,1], mat[:,3], udm(mat[:,5])
		elif direction == 'z':
			elon, dir2, dir3, s = mat[:,3], mat[:,1], mat[:,2], udm(mat[:,6])
		else:
			print('Houston we have a problem')
			exit()
		A = dir2 * dir3
		Stress = s * (A / A[0])
		Strain = (elon - elon[0]) / elon[0]
		Area = np.trapz(Stress, Strain)
		print('#Strain     Stress', file = OUT)
		
		for i in range(len(Stress)):
			print(Strain[i], Stress[i], file = OUT)
		OUT.close
		# curve fit
		popt, _ = curve_fit(objective, Strain, Stress, p0= [1000,-100,100,0], maxfev=5000)
		# summarize the parameter values and calculate the error
		# print a summary and a data file
		a, b, c, d = popt
		
		fig, ax = plt.subplots()
		ax.plot(Strain, Stress, c='b', label='MD')
		fitted = objective(Strain , a, b, c, d)
		ax.plot(Strain, fitted, c='r', label='Fit')
		title = 'Stress vs Strain on ' + direction + ' axis, using ' + str(final_step) + ' steps'
		ax.set_title(title)
		ax.set_xlabel('Strain')
		ax.set_ylabel('Stress')
		ax.legend()
		figure_name = 'StressStrain_' + direction + '_' + str(final_step)
		plt.savefig(figure_name, dpi=300)
				
		print('Fitting function on %s direction:' %direction)
		print('Stress = %.10E x^3 + %.10E x^2 + %.10E x + %.10E' % (a, b, c, d))
		print('**************************************')
		print('Young modulus in %s direction is: ' %direction)
		print('Area under Stress-Strain curve on %s direction is: ' %direction)
		print(c)
		
		plt.show()
		break
	else:
		print("Insert a direction:")
		print("You can use x, y, or z to calculate the Young modulus on that direction")
		print("Type 'all' to calculate the average Young modulus")



	
			
