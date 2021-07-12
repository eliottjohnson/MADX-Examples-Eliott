from scipy.interpolate import RegularGridInterpolator as RGI
import numpy as np
import math as m
import pandas as pd
# from sympy import Point3D, Plane, Line3D
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cmx
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib import colors

# constants
e_C = 1.602176634e-19     # unit electric charge [C]
c_m_per_s = 299792458.0   # speed of light [m/s]
mp_kg = 1.6726219e-27     # proton mass [kg]

def get_ang_from_mom(Px=0, Py=0, Pz=0):
    """ return angle from momentum """
    P = np.sqrt(Px**2 + Py**2 + Pz**2)
    return np.arctan(Px/Pz), np.arcsin(Py/P)

def get_mom_from_ang(xp, yp, P):
    """ return momentum form angle """
    Pz = P * np.sqrt((1 - np.sin(yp)**2) / (1 + np.tan(xp)**2))
    Px = Pz * np.tan(xp)   
    Py = P * np.sin(yp)
    return Px, Py, Pz

def get_can_from_ang(xp=0, yp=0, pz=0):
    """ return canonical momentum from angle """
    return (pz + 1) * np.sin(xp) * np.cos(yp), (pz + 1) * np.sin(yp)

def get_ang_from_can(px=0, py=0, pz=0):
    """ return angle from canonical momentum """
    xp = np.arctan(px / np.sqrt((pz + 1)**2 - px**2 - py**2))
    yp = np.arctan(py / np.sqrt((pz + 1)**2 - py**2)) 
    return xp, yp

def get_can_from_mom(Px, Py, Pz, P0):
    """ return canonical momentum from momentum """
    return Px / P0, Py / P0, (np.sqrt(Px**2 + Py**2 + Pz**2) - P0) / P0

def get_mom_from_can(px, py, pz, P0):
    """ return momentum from canonical momentum """
    Px = px * P0
    Py = py * P0
    Pz = np.sqrt(P0**2 * (pz + 1)**2 - Px**2 - Py**2)
    return Px, Py, Pz

def interpolate_fieldmap(df, method='linear'):
    """ interpolate the field map on a 3D regular grid
    and determine the boundaries of the field grid """
    df.sort_values(by=['x', 'y', 'z'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    xs, ys, zs = df.x.unique(), df.y.unique(), df.z.unique()
    Bx = RGI((xs, ys, zs), df.Bx.values.reshape((len(xs), len(ys), len(zs))), 
             bounds_error=False, fill_value=None, method=method)
    By = RGI((xs, ys, zs), df.By.values.reshape((len(xs), len(ys), len(zs))), 
             bounds_error=False, fill_value=None, method=method)
    Bz = RGI((xs, ys, zs), df.Bz.values.reshape((len(xs), len(ys), len(zs))), 
             bounds_error=False, fill_value=None, method=method)
    return {'Bx': Bx,
            'By': By,
            'Bz': Bz,
            'xbounds': ((df.x.min(), df.x.max()), (df.y.min(), df.y.max()), (df.z.min(), df.z.max()))}

def matrix_rotation(xp, yp, zp=0):
    """ return rotation matrix based on rotation angles around x, y, z axes """
    def Rx(theta):
        return np.array([[ 1, 0           , 0           ],
                         [ 0, m.cos(theta),-m.sin(theta)],
                         [ 0, m.sin(theta), m.cos(theta)]])
    def Ry(theta):
        return np.array([[ m.cos(theta), 0, m.sin(theta)],
                         [ 0           , 1, 0           ],
                         [-m.sin(theta), 0, m.cos(theta)]])
    def Rz(theta):
        return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                         [ m.sin(theta), m.cos(theta) , 0 ],
                         [ 0           , 0             , 1]])
    return np.dot(Rx(yp), Ry(-xp))

def get_pos_glob(pos, pos_glob, R):
    """ convert position pos in LCS to GCS
    based on the global position and the rotation matrix """
    return np.dot(np.linalg.inv(R), pos) +  pos_glob

def get_mom_glob(mom, R):
    """ convert momentum mom in LCS to GCS
    based on the rotation matrix """
    mom = np.array(mom)
    return np.dot(np.linalg.inv(R), mom)

def matrix_arr_to_dict(mat):
    """ convert 6x6 matrix from numpy array to dictionary """
    [[r11, r12, r13, r14, r15, r16],
    [r21, r22, r23, r24, r25, r26],
    [r31, r32, r33, r34, r35, r36],
    [r41, r42, r43, r44, r45, r46],
    [r51, r52, r53, r54, r55, r56],
    [r61, r62, r63, r64, r65, r66]] = mat
    
    return {'r11': r11, 'r12': r12, 'r13': r13, 'r14': r14, 'r15': r15, 'r16': r16, 
            'r21': r21, 'r22': r22, 'r23': r23, 'r24': r24, 'r25': r25, 'r26': r26, 
            'r31': r31, 'r32': r32, 'r33': r33, 'r34': r34, 'r35': r35, 'r36': r36, 
            'r41': r41, 'r42': r42, 'r43': r43, 'r44': r44, 'r45': r45, 'r46': r46, 
            'r51': r51, 'r52': r52, 'r53': r53, 'r54': r54, 'r55': r55, 'r56': r56, 
            'r61': r61, 'r62': r62, 'r63': r63, 'r64': r64, 'r65': r65, 'r66': r66}

def perpendicular_line(x1, y1, x2, y2):
    """ return a line perpendicular to the one that contains the two points (x1, y1) and (x2, y2) """
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1     
    return a, b 

def cross3(a, b):
    """ return cross product between two 3D vectors """
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])

def plane(vector, point):
    """ return plane normal to the vector """
    return [vector[0], vector[1], vector[2], -(vector[0]*point[0]+vector[1]*point[1]+vector[2]*point[2])]

def line_param(point1, point2):
    """ return coefficients of the parametric equation of line between two points """
    lx = point2[0] - point1[0]
    ly = point2[1] - point1[1]
    lz = point2[2] - point1[2]
    return lx, point1[0], ly, point1[1], lz, point1[2]

def point(line_par, plane):
    """ return intersection point between line and the plane """
    (lx, cx, ly, cy, lz, cz) = line_par
    (a,b,c,d) = plane
    t = -(cx*a + cy*b + cz*c + d)/(a*lx + b*ly + c*lz)
    return lx*t + cx, ly*t + cy, lz*t + cz

def distance( point1, point2):
    """ return distance between two points """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def remove_mid_el(arr):
    """ remove the middle element from an array """
    mid_id = round(len(arr)/2)
    return np.concatenate((arr[:mid_id], arr[mid_id+1:]))

def style_df_transport_matrix(arr,
                              caption=None,
                              index_names = ["x [m]","xp [rad]","y [m]","yp [rad]","T [m]","D [1]"],
                              columns_names = ["x0 [m]","xp0 [rad]","y0 [m]","yp0 [rad]","T0 [m]","D0 [1]"]):
    df = pd.DataFrame(data = arr,
                      index = index_names,
                      columns = columns_names)
    if caption:
        return df.style.format("{:.4f}").set_caption(caption)
    return df.style.format("{:.4f}")


class Particle(object):
    def __init__(self, name):
        self.partd = {'proton' : {'e0': 9.3827208816e8,
                                  'a' : 1.0,
                                  'z' : 1.0},
                      'carbon12': {'e0': ((9.38272*6+9.395654*6)-0.93161)*1e8,
                                   'a' : 12.0,
                                   'z' : 6.0 }}
        assert name in self.partd
        self.particle = self.partd[name]
        self.a = self.particle['a']
        self.z = self.particle['z']
        self.e0_eV = self.particle['e0']

    def get_e0_eV(self):
        """ return rest energy of the particle in eV """
        return self.e0_eV

    def get_e0_MeV(self):
        """ return rest energy of the particle in MeV """
        return self.e0_eV / 1e6
    
    def get_a(self):
        """ return mass number """
        return self.a

    def get_z(self):
        """ return atomic number """
        return self.z
    
    def get_en_tot(self, en_per_unit_MeV):
        """ return total energy of particle in eV """
        return self.e0_eV + self.a * en_per_unit_MeV * 1e6

    def get_p0c(self, en_per_unit_MeV):
        """ return particle momentum in eV """
        etot_eV = (en_per_unit_MeV * 1e6 + self.e0_eV)
        return np.sqrt(self.get_en_tot(en_per_unit_MeV)**2 - self.e0_eV**2)

    def get_en_per_unit_MeV(self, p0c):
        """ return energy per unit in MeV based on momentum """
        res_ev = np.sqrt(p0c**2 + self.e0_eV**2) - self.e0_eV
        return res_ev * 1e-6 / self.a

    def get_en_per_unit_MeV_rig(self, rigidity_Tm):
        """ return energy per unit in MeV based on rigidity T-m """
        momentum = rigidity_Tm * e_C * self.z # kg.m/s
        e_tot_J = c_m_per_s * np.sqrt(momentum**2 + (self.a * mp_kg * c_m_per_s)**2) # J
        e_tot = e_tot_J / e_C # eV
        e_kin = e_tot - self.e0_eV * 1e6
        return e_kin / self.a / 1e6  

    def get_rigidity(self, en_per_unit_MeV):
        """ return beam rigidity in T-m """
        e_kin = self.a * en_per_unit_MeV * 1e6 
        e_tot = self.e0_eV + e_kin # eV
        e_tot_J = e_tot * e_C # J
        momentum = np.sqrt((e_tot_J / c_m_per_s)**2 - (self.a * mp_kg * c_m_per_s)**2) # kg.m/s
        return momentum / (e_C * self.z)

    def get_lorentz_beta(self, en_per_unit_MeV):
        """ return the Lorentz beta factor based on energy per unit in MeV"""
        gamma = en_per_unit_MeV / self.get_e0_MeV() + 1
        return np.sqrt(1 - ((1 / gamma)**2))


class SetGenerator(object):
    """ generate a set of particles for the tracker """
    
    def __init__(self, particleName):
        assert particleName.lower() in ['proton', 'carbon12'], 'wrong particle name'
        self.particle = Particle(particleName)
        self.e0_eV = self.particle.get_e0_eV()
        self.a = self.particle.get_a()
        self.z = self.particle.get_z()
        self.m0_MeV = self.particle.get_e0_MeV()

    def get_part1_local(self, distr_part_ref, var_name, dvar_value, pid):
        """ distr_part_ref is of type dictionary with fields:
            dX, dXP, dY, dYP, dS, en_MeV, dD, dt.
            var_name is a string of one of above
            dvar_value is the change value applied to var_name field """
        distr = distr_part_ref.copy() # copy the reference particle into a new one
        distr['pid'] = pid
        if var_name == 'dD':
            mom0 = self.particle.get_p0c(distr_part_ref['en_MeV']) * (1 + dvar_value)
            distr['en_MeV'] = self.particle.get_en_per_unit_MeV(mom0)
        else:
            distr[var_name] += dvar_value
        return distr
    
    def get_part1_global(self, distr, pos_glob0, ang_glob0, **kwargs):
        """ take distribution (distr) of one particle in local coord. system
        and convert to global coord. system """
        # read out local angles/energy and convert into momentum
        P_loc = get_mom_from_ang(xp = distr['dXP'],
                                 yp = distr['dYP'],
                                 P = self.particle.get_p0c(distr['en_MeV']))
        # move all to global
        R = matrix_rotation(*ang_glob0)
        # position to global
        pos_glob = get_pos_glob(pos = np.array([distr['dX'], distr['dY'], distr['dS']]),
                                   pos_glob = pos_glob0,
                                   R = R)
        # momentum to global
        mom_glob = get_mom_glob(mom = list(P_loc),
                                   R = R)
        direction = kwargs.pop("direction", None)
        if direction is None:
            return distr['pid'], pos_glob, mom_glob, distr['dt']
        else:
            return distr['pid'], pos_glob, mom_glob, distr['dt'], direction
    
    def get_partset_global(self, set_local, pos_glob0, ang_glob0, direction=None):
        """ return the whole set of particles in GCS
        based on the set in LCS """
        set_global = []
        for key, value in set_local.items():
            set_global.append(self.get_part1_global(distr = value, pos_glob0 = pos_glob0, ang_glob0 = ang_glob0, direction=direction))
        return set_global

    def get_part13_local(self, distr_part_ref, dX, dXP, dY, dYP, dt, dD):
        """ local information on 13 particles based on offsets and reference particle """
        n_part = 13
        names = ['dX', 'dX', 'dXP', 'dXP', 'dY', 'dY', 'dYP', 'dYP', 'dt', 'dt', 'dD', 'dD']
        vals = [dX, -dX, dXP, -dXP, dY, -dY, dYP, -dYP, dt, -dt, dD, -dD]
        distr13 = {}
        for i in range(n_part):
            distr13[i] = distr_part_ref.copy()
            distr13[i]['pid'] = i
            if i > 0:
                distr13[i] = self.get_part1_local(distr_part_ref = distr_part_ref,
                                                  var_name = names[i-1],
                                                  dvar_value = vals[i-1],
                                                  pid = i)
        return distr13


class Tracks(object):
    """ work on the set of tracks (list of dataframes, each dataframe corresponds to the track of one particle) """
    
    def __init__(self, tracks_set, particle, ref_pid=0):
        self.tracks_set = tracks_set
        self.track_ref = tracks_set[ref_pid]
        self.part_name = Particle(particle)

    def set_ref_pid(self, ref_pid):
        """ set the id of the reference particle and the reference track """
        self.ref_pid = ref_pid
        self.track_ref = self.tracks_set[ref_pid]
       
    def get_ref_last_k(self):
        """ get the last step of the reference particle track """
        return self.track_ref.shape[0]-1
    
    def get_pos_k_global(self, k):
        """ return position components of reference particle at point k """
        return np.array([self.track_ref.iloc[k]['x'], self.track_ref.iloc[k]['y'], self.track_ref.iloc[k]['z']])

    def get_mom_k_global(self, k):
        """ return momentum components of reference particle at point k """
        return np.array([self.track_ref.iloc[k]['Px'], self.track_ref.iloc[k]['Py'], self.track_ref.iloc[k]['Pz']])

    def get_p0(self, k):
        """ return momentum scalar of reference particle at point k """
        return np.sqrt((self.get_mom_k_global(k)**2).sum())

    def get_system_rotation_matrix(self, k):
        """ return rotation matrix based on the momentum of reference particle at point k """
        mom_k = self.get_mom_k_global(k)
        return matrix_rotation(*get_ang_from_mom(*mom_k))

    def get_pos_loc(self, pos, pos_ref, k):
        """ convert position pos in GCS to LCS
        based on the local position of a particle at point k 
        and the local position of the reference particle at point k """
        R = self.get_system_rotation_matrix(k)
        return np.dot(R, pos.T - pos_ref.T)
    
    def get_mom_loc(self, mom, k):
        """ convert position mom in GCS to LCS at point k """
        R = self.get_system_rotation_matrix(k)
        return np.dot(R, mom.T)

    def get_part_at_k(self, pid, k):
        """ return exact information of the particle of id pid at point k in GCS """
        tref_shape = self.track_ref.shape[0]
        assert k <= tref_shape, 'There is only {} steps in the reference particle track'.format(tr_shape)

        df = self.tracks_set[pid]
        pos_k = self.get_pos_k_global(k)
        df['distr_ref'] = np.sqrt((df['x'] - pos_k[0])**2 + (df['y'] - pos_k[1])**2 + (df['z'] - pos_k[2])**2)
        dfsort = df.sort_values(by=['distr_ref'])

        # the first closest track point to the reference particle track point for each particle 
        pos_c1 = np.array([float(dfsort.x.iloc[0]), float(dfsort.y.iloc[0]), float(dfsort.z.iloc[0])])
        mom_c1 = np.array([float(dfsort.Px.iloc[0]), float(dfsort.Py.iloc[0]), float(dfsort.Pz.iloc[0])])
        t1 = float(dfsort.t.iloc[0]) # time
        
        return pos_c1, mom_c1, t1
    
    def get_part_at_k_int(self, pid, k):
        """ return approximated particle information at point k in global coord. system """
        tref_shape = self.track_ref.shape[0]
        assert k <= tref_shape, 'There is only {} steps in the reference particle track'.format(tr_shape)

        # reference particle @ point k: position and momentum at this point
        df = self.tracks_set[pid]
        pos_k = self.get_pos_k_global(k)
        mom_k = self.get_mom_k_global(k)

        df = self.tracks_set[pid]
        df['distr_ref'] = np.sqrt((df['x'] - pos_k[0])**2 + (df['y'] - pos_k[1])**2 + (df['z'] - pos_k[2])**2)
        dfsort = df.sort_values(by=['distr_ref'])

        # plane normal to the reference vector
        plane_k = plane(vector = mom_k, point = pos_k)
        
        # the first closest (c1) and second closest (c2) track point to the reference particle track point for each particle 
        # lines between c1 and c2
        pos_c1 = np.array([float(dfsort.x.iloc[0]), float(dfsort.y.iloc[0]), float(dfsort.z.iloc[0])])
        pos_c2 = np.array([float(dfsort.x.iloc[1]), float(dfsort.y.iloc[1]), float(dfsort.z.iloc[1])])
        line_c1c2 = line_param(point1 = pos_c1, point2 = pos_c2)

        # intersection point between line (c1c2) and the plane_k (normal to the reference traj)
        ip = point(line_par = line_c1c2, plane = plane_k)

        # global position on the reference plane
        pos_g = np.array([ip[0], ip[1], ip[2]])
        #print('pos_c1 = {}, pos_c2 = {}, pos_g = {}'.format(pos_c1, pos_c2, pos_g))

        # distances of the intersection point to the previous and following points in the track
        # they will be weights to determine the momentum at intersection point
        d1 = distance( ip, pos_c1)
        d2 = distance( ip, pos_c2)
        w1 = d1 / (d1 + d2)
        w2 = d2 / (d1 + d2)

        # global momentum at points c1 and c2
        mom_c1 = np.array([float(dfsort.Px.iloc[0]), float(dfsort.Py.iloc[0]), float(dfsort.Pz.iloc[0])])
        mom_c2 = np.array([float(dfsort.Px.iloc[1]), float(dfsort.Py.iloc[1]), float(dfsort.Pz.iloc[1])])
        mom_c1c2 = np.array([mom_c1, mom_c2]).T

        mom_g = np.average(mom_c1c2, axis=1, weights=[w1, w2])
        #print('mom_c1 = {}, mom_c2 = {}, mom_g = {}'.format(mom_c1, mom_c2, mom_g))

        # time
        t1 = float(dfsort.t.iloc[0])
        t2 = float(dfsort.t.iloc[1])
        ttime = np.average([t1, t2],  weights=[w1, w2])
        print('w1 = {}, w2 = {}, t1 = {}, t2 = {}, ttime = {}'.format(w1, w2, t1, t2, ttime))

        return pos_g, mom_g, t1

    def get_tracks_set_loc(self, k):
        """ return the set of tracks in the LCS of the reference particle at point k """
        tref_shape = self.track_ref.shape[0]
        assert k <= tref_shape, 'There is only {} steps in the reference particle track'.format(tr_shape)
    
        # reference particle @ point k: position and momentum at this point
        pos_k = self.get_pos_k_global(k)
        mom_k = self.get_mom_k_global(k)
        p0_k = self.get_p0(k)

        # position and momentum in the reference system of ref. particle
        pos_l, mom_l, ang_l, p0, D, t = [], [], [], [], [], []
        for pid in range(0, len(self.tracks_set)):
            # approximate particle information at point k in global coord. system
            pos_g, mom_g, t_g = self.get_part_at_k(pid=pid, k=k)
            #pos_g, mom_g, t_g = self.get_part_at_k_int(pid=pid, k=k)
     
            # convert the approximated values to local coordinate system of reference particle
            pos_l.append(self.get_pos_loc(pos_g, pos_k, k))
            mom_l_ = self.get_mom_loc(mom_g, k)
            mom_l.append(mom_l_)
            xp_l, yp_l = get_ang_from_mom(*mom_l_)
            ang_l.append(np.array([xp_l, yp_l]))
            p0.append(np.sqrt(mom_g[0]**2 + mom_g[1]**2 +mom_g[2]**2 ))
            D.append((p0[pid]-p0_k)/p0_k)
            t.append(t_g)
        
        return {'pos_l': pos_l, 'mom_l': mom_l, 'ang_l': ang_l, 'p0': p0, 'D': D, 't': t}

    def get_transport_matrix(self, k, ret='mat'):
        """ return the 1st order 6x6 transport matrix from the beginning to the point k """
        input_res = self.get_tracks_set_loc(0) # for initial offsets

        dX = input_res['pos_l'][1][0] - input_res['pos_l'][0][0]
        dXP = input_res['ang_l'][3][0] - input_res['ang_l'][0][0]
        dY = input_res['pos_l'][5][1] - input_res['pos_l'][0][1]
        dYP = input_res['ang_l'][7][1] - input_res['ang_l'][0][1]
        dT = -c_m_per_s*(input_res['t'][9] - input_res['t'][0])
        dD = input_res['D'][11] - input_res['D'][0]        
        #print('Deltas = {}'.format((dX, dXP, dY, dYP, dT/c_m_per_s, dD)))
        
        # at point k
        output_res = self.get_tracks_set_loc(k)
        T = []
        for pid in range(0, len(self.tracks_set)):
            T.append(-c_m_per_s*(output_res['t'][pid] - output_res['t'][0]))

        r11 = (output_res['pos_l'][1][0] - output_res['pos_l'][2][0]) / 2 / dX
        r12 = (output_res['pos_l'][3][0] - output_res['pos_l'][4][0]) / 2 / dXP
        r21 = (output_res['ang_l'][1][0] - output_res['ang_l'][2][0]) / 2 / dX
        r22 = (output_res['ang_l'][3][0] - output_res['ang_l'][4][0]) / 2 / dXP
        
        r13 = (output_res['pos_l'][5][0] - output_res['pos_l'][6][0]) / 2 / dY
        r14 = (output_res['pos_l'][7][0] - output_res['pos_l'][8][0]) / 2 / dYP
        r23 = (output_res['ang_l'][5][0] - output_res['ang_l'][6][0]) / 2 / dY
        r24 = (output_res['ang_l'][7][0] - output_res['ang_l'][8][0]) / 2 / dYP
        
        r15 = (output_res['pos_l'][9][0] - output_res['pos_l'][10][0]) / 2 / dT
        r16 = (output_res['pos_l'][11][0] - output_res['pos_l'][12][0]) / 2 / dD
        r25 = (output_res['ang_l'][9][0] - output_res['ang_l'][10][0]) / 2 / dT
        r26 = (output_res['ang_l'][11][0] - output_res['ang_l'][12][0]) / 2 / dD
        
        r31 = (output_res['pos_l'][1][1] - output_res['pos_l'][2][1]) / 2 / dX
        r32 = (output_res['pos_l'][3][1] - output_res['pos_l'][4][1]) / 2 / dXP
        r41 = (output_res['ang_l'][1][1] - output_res['ang_l'][2][1]) / 2 / dX
        r42 = (output_res['ang_l'][3][1] - output_res['ang_l'][4][1]) / 2 / dXP
        
        r33 = (output_res['pos_l'][5][1] - output_res['pos_l'][6][1]) / 2 / dY
        r34 = (output_res['pos_l'][7][1] - output_res['pos_l'][8][1]) / 2 / dYP
        r43 = (output_res['ang_l'][5][1] - output_res['ang_l'][6][1]) / 2 / dY
        r44 = (output_res['ang_l'][7][1] - output_res['ang_l'][8][1]) / 2 / dYP
        
        r35 = (output_res['pos_l'][9][1] - output_res['pos_l'][10][1]) / 2 / dT
        r36 = (output_res['pos_l'][11][1] - output_res['pos_l'][12][1]) / 2 / dD
        r45 = (output_res['ang_l'][9][1] - output_res['ang_l'][10][1]) / 2 / dT
        r46 = (output_res['ang_l'][11][1] - output_res['ang_l'][12][1]) / 2 / dD

        r51 = (T[1] - T[2]) / 2 / dX
        r52 = (T[3] - T[4]) / 2 / dXP
        r61 = (output_res['D'][1] - output_res['D'][2]) / 2 / dX
        r62 = (output_res['D'][3] - output_res['D'][4]) / 2 / dXP

        r53 = (T[5] - T[6]) / 2 / dY
        r54 = (T[7] - T[8]) / 2 / dYP
        r63 = (output_res['D'][5] - output_res['D'][6]) / 2 / dY
        r64 = (output_res['D'][7] - output_res['D'][8]) / 2 / dYP
        
        r55 = (T[9] - T[10]) / 2 / dT
        r56 = (T[11] - T[12]) / 2 / dD
        r65 = (output_res['D'][9] - output_res['D'][10]) / 2 / dT
        r66 = (output_res['D'][11] - output_res['D'][12]) / 2 / dD

        mat = [[r11, r12, r13, r14, r15, r16],
               [r21, r22, r23, r24, r25, r26],
               [r31, r32, r33, r34, r35, r36],
               [r41, r42, r43, r44, r45, r46],
               [r51, r52, r53, r54, r55, r56],
               [r61, r62, r63, r64, r65, r66]]
      
        if ret == 'mat':
            return mat
        elif ret == 'mat_distance':
            pos_g, _, _ = self.get_part_at_k(0, k)
            return mat, pos_g
        
    def get_madx_matrix(self, mat_arr, k):
        """ return the transport matrix in the typical madx format 
        based on the transport matrix mat_arr at point k """
        madx_mat_arr = np.copy(mat_arr)

        # get local outputs at point k
        output_res = self.get_tracks_set_loc(k)
        can_l, T, PT = [], [], []
        for pid in range(0, len(self.tracks_set)):
            can_l.append(get_can_from_mom(output_res['mom_l'][pid][0],
                                          output_res['mom_l'][pid][1],
                                          output_res['mom_l'][pid][2],
                                          output_res['p0'][pid]))
            T.append(-c_m_per_s * (output_res['t'][pid] - output_res['t'][0]))
            lbeta = self.part_name.get_lorentz_beta(self.part_name.get_en_per_unit_MeV(output_res['p0'][pid]))
            PT.append(lbeta * output_res['D'][pid])

        # recalculate coefficients
        for row in range(0,6):
            # column px
            madx_mat_arr[row][1] = madx_mat_arr[row][1] * output_res['ang_l'][3][0] / can_l[3][0]         
            # column py
            madx_mat_arr[row][3] = madx_mat_arr[row][3] * output_res['ang_l'][7][1] / can_l[7][1]   
            # column PT
            madx_mat_arr[row][5] = madx_mat_arr[row][5] * output_res['D'][11] / PT[11] 

        for col in range(0,6):
            # row px
            madx_mat_arr[1][col] = madx_mat_arr[1][col] * can_l[3][0] / output_res['ang_l'][3][0] 
            # row py
            madx_mat_arr[3][col] = madx_mat_arr[3][col] * can_l[7][1] / output_res['ang_l'][7][1] 
            # row PT
            madx_mat_arr[5][col] = madx_mat_arr[5][col] * PT[11] / output_res['D'][11] 

        return madx_mat_arr

    def get_track_condition(self):
        """ return coeffcients of the line that determines the end of tracking for a set of particles
        based on a previously run single reference particle track """
        z1 = self.track_ref['z'].iloc[-1]
        x1 = self.track_ref['x'].iloc[-1]
        z2 = self.track_ref['z'].iloc[-2]
        x2 = self.track_ref['x'].iloc[-2]
        a_ref, b_ref = perpendicular_line(z1,x1,z2,x2)
        a = -1/a_ref
        b = x1 - a*z1
    
        return (a,b)
    
    def plot_track(self, pid, legend = False, **kwargs):
        """ plot track of the particle pid """
        track = self.tracks_set[pid]
        
        label = kwargs.pop("label", None)
        if label is None:
            label = "particle " + str(pid)
            
        figsize = kwargs.pop("figsize", None)
        if figsize is None:
            figsize=(5,5)
        
        axes = kwargs.pop("axes", None)
        plot_yz = kwargs.pop("plot_yz", False)
        if plot_yz is False:
            
            if axes is None:
                fig, axes = plt.subplots(2, 1, figsize=figsize, tight_layout=True, sharex=False)
            [ax0, ax1] = axes
            
            ax0.plot(track['z'], track['y'], label = label)
            ax0.set_ylabel('y [m]')
            ax1.plot(track['z'], track['x'], label = label)
            ax1.set_ylabel('x [m]')
            ax1.set_xlabel('z [m]')
            if legend:
                ax0.legend()
                ax1.legend()
            return axes
            
        else:
            if axes is None:
                fig, ax0 = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
            else:
                ax0 = axes
            ax0.plot(track['z'], track['y'], label = label)
            ax0.set_ylabel('y [m]')
            if legend:
                ax0.legend()            
            return ax0
    
    def save_ref_to_csv(self, filename):
        """ save the reference particle trajectory to a csv file """
        ref_trajectory = self.track_ref.copy()
        ref_trajectory = ref_trajectory.drop(columns=['id', 'k', 'Px', 'Py', 'Pz', 't'])
        ref_trajectory.to_csv(filename, index = False)


class Fitter(object):
    """ polynomial fits to a 2D set of points """ 
    
    def __init__(self, xt, yt):
        self.xt = xt
        self.yt = yt
    
    def get_fit(self, order):
        """ return the polynomial fit of the given order """
        return np.polyfit(self.xt, self.yt, order)
    
    def get_y_fitted(self, order):
        """ return the fitted values """
        p = self.get_fit(order)
        return np.polyval(p, self.xt)
    
    def get_residuals(self, order):
        """ return the residuals between the inputs and fitted values """
        yf = self.get_y_fitted(order)
        return self.yt - yf
    
    def get_rel_errors(self, order):
        """ return the relative errors of the fitted values """
        res = self.get_residuals(order)
        return res / self.yt
