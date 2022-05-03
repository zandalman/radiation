import os, sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from npy_append_array import NpyAppendArray

# constants

h = 6.6260755e-27 # Plank's constant [erg / s]
sigma_T = 6.6524587e-25 # Thompson cross section [cm^2]
m_p = 1.6733e-24 # Hydrogen mass [g]

# event IDs

START = 0
SCATTER = 1
ABSORB = 2
TERMINATE = 3
MAX = 4


def norm(vec):
  ''' Compute the of a vector array with components along the first axis '''
  return np.sqrt(np.sum(vec**2, axis=0))


class Grid(object):
  '''
  Class for grid

  Parameters
  x_min, x_max (float): Bounds of the grid in the x direction
  y_min, y_max (float): Bounds of the grid in the y direction
  z_min, z_max (float): Bounds of the grid in the z direction
  res (int): Number of cells in each dimension

  Attributes

  x_min, x_max (float): Bounds of the grid in the x direction
  y_min, y_max (float): Bounds of the grid in the y direction
  z_min, z_max (float): Bounds of the grid in the z direction
  res (int): Number of cells in each dimension

  dx (array): Grid spacing in each dimension; (d) array
  epsilon (array): Small numerical factor relative to dx in each dimension; (d) array

  x1d (array): Cell centerpoints in each dimension; (d x M) array
  x (array): Cell centerpoints; (d x M x M x M) array
  r (array): Norm of cell centerpoints; (M x M x M) array

  X (float): Mass fraction of hydrogen
  Y (float): Mass fraction of helium
  Z (float): Mass fraction of metals

  rho (array): Density; (M x M x M) array
  n_e (array): Number density of electrons; (M x M x M) array
  '''
  def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max, res):

    self.x_min = x_min
    self.x_max = x_max
    self.y_min = y_min
    self.y_max = y_max
    self.z_min = z_min
    self.z_max = z_max
    self.res = res

    self.dx = np.array([self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min]) / (self.res - 1)
    self.epsilon = self.dx / 1e6

    self.x1d = np.array([np.linspace(self.x_min, self.x_max, self.res), np.linspace(self.y_min, self.y_max, self.res), np.linspace(self.z_min, self.z_max, self.res)])
    self.x = np.array(np.meshgrid(self.x1d[0], self.x1d[1], self.x1d[2]))
    self.r = norm(self.x)

    self.X = 0.7
    self.Y = 0.28
    self.Z = 0.02

    self.rho = np.zeros_like(self.r)

  @property
  def n_e(self):
    ''' Compute the electron number density '''
    return self.rho / (2 * m_p) * (self.X + 1)


class Sim(object):
  '''
  Class for simulations.

  Parameters

  grid (object): Grid object
  N (int): Number of photons
  albedo (float): Albedo of the electron gas
  do_rayleigh (bool): Use Rayleigh scattering; if false, use isotropic scattering

  seed (int): Seed for the random number generator; if None, seed is chosen randomly
  outfile_path (str): Path of the output data file
  log_path (str): Path of the output log file

  Attributes

  grid (object): Grid object

  N (int): Number of packets
  albedo (float): Albedo of the electron gas

  t (array): Time of each packet; (N) array
  pos (array): Coordinates of each packet; (d x N) array
  cos_th, phi (array): Pointing of each packet in spherical coordinates; (N) array
  pointing (array): Poiniting of each packet in Cartesian coordinates; (d x N) array

  tau_run (array): Running optical depth of each packet; (N) array
  tau (array): Optical depth of each packet; (N) array

  d_flag (array): Stop evolution flag for each packet; (N) array
  i_flag (array): Interaction flag for each packet; (N) array
  n_flag (array): No interaction flag for each packet; (N) array
  s_flag (array): Scattering flag for each packet; (N) array
  a_flag (array): Absorption flag for each packet; (N) array

  S (array): Stokes vector (S, Q, U, V) excluding the circular polarization V; (3 x N) array
  do_rayleigh (bool): Use Rayleigh scattering; if false, use isotropic scattering

  rng (object): Random number generator
  random (array): Random numbers uniformely sampled over interval [0, 1]; (N) array
  nsteps (int): Number of simulation steps

  outfile_path (str): Path of the output data file
  log_path (str): Path of the output log file
  '''
  def __init__(self, grid, N=1, do_rayleigh=True, seed=None, albedo=1, outfile_path='data/out.npy', log_path='logs/out.log'):

    self.grid = grid

    self.N = N
    self.albedo = albedo

    self.t = np.zeros((self.N))
    self.pos = np.zeros((3, self.N))
    self.cos_th, self.phi = np.zeros((self.N)), np.zeros((self.N))
    self.tau_run = np.zeros((self.N))
    self.tau = np.zeros((self.N))

    self.d_flag = np.zeros((self.N), dtype=bool)
    self.i_flag = np.zeros((self.N), dtype=bool)
    self.n_flag = np.zeros((self.N), dtype=bool)
    self.s_flag = np.zeros((self.N), dtype=bool)
    self.a_flag = np.zeros((self.N), dtype=bool)

    self.S = np.zeros((3, self.N))
    self.S[0] = 1

    self.do_rayleigh = do_rayleigh

    self.rng = np.random.default_rng(seed)
    self.nsteps = 0

    self.init_photons()

    # create outfile
    if os.path.isfile(outfile_path): os.remove(outfile_path)
    self.outfile_path = outfile_path

    # create log
    if os.path.isfile(log_path): os.remove(log_path)
    self.log_path = log_path
    self.log("Process ID: %d" % os.getpid()) # log process ID

    # write initial conditions to outfile
    self.write_photons(START, np.ones((self.N), dtype=bool))

  def log(self, message):
      '''
      Log a message to the log file.

      Parameters
      message (str): Message to log
      '''
      with open(self.log_path, 'a') as f:
          f.write(message)
          f.write('\n')

  def init_photons(self):
    ''' Initialize packet positions and pointings. '''
    self.cos_th[:], self.phi[:] = np.sqrt(self.random), 2 * np.pi * self.random
    self.pos[2] = -7

  def write_photons(self, event_id, flag):
    '''
    Write data about packets undergoing events.
    Specifically, for each event we write
      event ID, packet index, time, position, pointing, Stokes vector.

    Parameters
    event_id (int): Identifier for the type of event
      START = 0; packet initial conditions
      SCATTER = 1; packet was scattered
      ABSORB = 2; packet was absorped
      TERMINATE = 3; end condition achieved
      MAX = 4; max steps reached
    flag (array): Event flag; (N) array
    '''
    event_id = np.full(np.sum(flag), event_id)
    id = np.where(flag)[0]
    t = self.t[flag]
    x, y, z = self.pos[:, flag]
    cos_th, phi = self.cos_th[flag], self.phi[flag]
    I, Q, U = self.S[:, flag]

    # create array
    arr = np.asfortranarray([event_id, id, t, x, y, z, cos_th, phi, I, Q, U]).T

    # append array to npy outfile
    with NpyAppendArray(self.outfile_path) as f:
      f.append(arr)

  def step_cell(self):
    ''' Step each packet by one cell. '''
    # compute distance to nearest cell boundary along pointing
    dis_to_cell = np.abs(self.grid.x1d[:, None, :] - self.pos[:, :, None]) # distances between each packet and each cell centerpoint in each dimension; (d x N x M) array
    cell_idx = np.argmin(dis_to_cell, axis=-1) # indices of cell containing each packet; (d x N) array
    home_cell_expanded = self.grid.x1d.T[cell_idx.T].T # coordinates of centerpoint of cell containing each packet with expanded dimension; (d x d x N) array
    home_cell = np.diagonal(home_cell_expanded, axis1=0, axis2=1).T # coordinates of centerpoint of cell containing each packet; (d x N) array
    neighbor_cells = np.copy(home_cell_expanded)
    neighbor_cells[np.diag_indices(3)] += np.sign(self.pointing) * self.grid.dx[:, None]
    neighbor_cells = np.swapaxes(neighbor_cells, 0, 1) # coordinates of centerpoints of cells neighboring cells containing each packet; (d x d x N) array
    vec_to_neighbors = neighbor_cells - self.pos[:, None] # vectors from packet to centerpoints of cells neighboring cell containing each packet; (d x d x N) array
    vec_to_home = home_cell - self.pos # vector from packet to cell centerpoint of cell containing each packet; (d x N) array
    vec_home_to_neighbor = vec_to_neighbors - vec_to_home[:, None] # vectors from centerpoints of cell containing each packet to centerpoints of neighboring cells; (d x d x N) array
    dis_to_boundary = (norm(vec_to_neighbors)**2 - norm(vec_to_home)**2) / (2 * np.sum(vec_home_to_neighbor * self.pointing, axis=0)) # distances to neighboring cells for each packet; (d x N) array
    dis_to_boundary_min = np.min(dis_to_boundary, axis=0) + self.grid.epsilon[np.argmin(dis_to_boundary, axis=0)] # distance to next cell for each packet; (N) array

    # get cell properties
    n_e_cell = grid.n_e[cell_idx[0], cell_idx[1], cell_idx[2]] # electron number density of cell containing each packet; (N) array
    tau_cell = sigma_T * n_e_cell * dis_to_boundary_min # increase in optical depth in cell for each packet; (N) packet

    # flag packets for interaction
    self.i_flag = ((self.tau_run + tau_cell) > self.tau) * np.logical_not(self.d_flag)
    self.n_flag = ((self.tau_run + tau_cell) < self.tau) * np.logical_not(self.d_flag)

    # update optical depths
    self.tau_run[self.i_flag] = 0 # reset running optical depth for interacting packets
    self.tau[self.i_flag] = -np.log(self.random)[self.i_flag] # resample optical depth for interacting packets
    self.tau_run[self.n_flag] += tau_cell[self.n_flag] # update running optical depth for non-interacting packets

    # update packet times
    self.t[self.i_flag] += ((self.tau - self.tau_run) / (sigma_T * n_e_cell))[self.i_flag] # for interacting packets
    self.t[self.n_flag] += dis_to_boundary_min[self.n_flag] # for non-interacting packets

    # update packet positions
    self.pos[:, self.i_flag] += ((self.tau - self.tau_run) / (sigma_T * n_e_cell) * self.pointing)[:, self.i_flag] # for interacting packets
    self.pos[:, self.n_flag] += (dis_to_boundary_min * self.pointing)[:, self.n_flag] # for non-interacting packets

  def interact(self):
    ''' Create interaction events. '''
    # flag packets for scattering and absorption depending on albedo
    self.s_flag = (self.random < self.albedo) * self.i_flag
    self.a_flag = (self.random > self.albedo) * self.i_flag

    # update pointing and Stokes vector for scattered packets
    self.write_photons(SCATTER, self.s_flag)
    if self.do_rayleigh:
      self.rayleigh()
    else:
      self.cos_th[self.s_flag] = 2 * self.random[self.s_flag] - 1
      self.phi[self.s_flag] = 2 * np.pi * self.random[self.s_flag]

    # stop evolution for absorped packets
    self.write_photons(ABSORB, self.a_flag)
    self.d_flag[self.a_flag] = True

  def rayleigh(self):
    '''
    Update pointing and Stokes vectors for Rayleigh scattering events.
    See Chandrasekhar (1960).
    '''
    I_new = np.zeros((self.N)) # new intensity; (N) array

    # sample i1 from a uniform distribution
    i1 = 2 * np.pi * self.random # angle from scattering geometry; (N) array
    i2 = np.zeros_like(i1) # angle from scattering geometry; (N) array

    # sample scattering angle Theta using the rejection method
    cos_Th = 1 - 2 * self.random # cosine of scattering angle; (N) array
    P1, P2 = cos_Th**2 + 1, cos_Th**2 - 1
    t_flag = np.zeros((self.N), dtype=bool) # try again flag for rejection method; (N) array
    t_flag[self.s_flag] = True
    while np.any(t_flag):
      cos_Th[t_flag] = 1 - 2 * self.random[t_flag]
      P1, P2 = cos_Th**2 + 1, cos_Th**2 - 1
      I_new[t_flag] = 3/4 * (P1[t_flag] * self.S[0, t_flag] + P2[t_flag] * np.cos(2 * i1[t_flag]) * self.S[1, t_flag] - P2[t_flag] * np.sin(2 * i1[t_flag]) * self.S[2, t_flag]) # update intensity
      t_flag[t_flag] = 3/2 * self.random[t_flag] * self.S[0, t_flag] > I_new[t_flag] # update t_flag

    # compute i2 and new pointing using spherical trigonometry
    sin_Th = np.sqrt(1 - cos_Th**2)
    sin_th_old = self.sin_th
    self.cos_th[self.s_flag] = (self.cos_th[self.s_flag] * cos_Th[self.s_flag] + self.sin_th[self.s_flag] * sin_Th[self.s_flag] * np.cos(i1[self.s_flag]))
    self.phi[self.s_flag] = (np.arcsin(sin_Th[self.s_flag] * np.sin(i1[self.s_flag]) / self.sin_th[self.s_flag]) + self.phi[self.s_flag])
    i2[self.s_flag] = np.arcsin(sin_th_old[self.s_flag] * np.sin(i1[self.s_flag]) / self.sin_th[self.s_flag])

    # compute new Stokes vector
    self.S[0, self.s_flag] = I_new[self.s_flag]
    self.S[1, self.s_flag] = 3/4 * (P2[self.s_flag] * np.cos(2 * i2[self.s_flag]) * self.S[0, self.s_flag] +
                      (P1[self.s_flag] * np.cos(2 * i1[self.s_flag]) * np.cos(2 * i2[self.s_flag]) - 2 * cos_Th[self.s_flag] * np.sin(2 * i1[self.s_flag]) * np.sin(2 * i2[self.s_flag])) * self.S[1, self.s_flag] +
                      (-P1[self.s_flag] * np.sin(2 * i1[self.s_flag]) * np.cos(2 * i2[self.s_flag]) - 2 * cos_Th[self.s_flag] * np.cos(2 * i1[self.s_flag]) * np.sin(2 * i2[self.s_flag])) * self.S[2, self.s_flag])
    self.S[2, self.s_flag] = 3/4 * (P2[self.s_flag] * np.sin(2 * i2[self.s_flag]) * self.S[0, self.s_flag] +
                      (P1[self.s_flag] * np.cos(2 * i1[self.s_flag]) * np.sin(2 * i2[self.s_flag]) + 2 * cos_Th[self.s_flag] * np.sin(2 * i1[self.s_flag]) * np.cos(2 * i2[self.s_flag])) * self.S[1, self.s_flag] +
                      (-P1[self.s_flag] * np.sin(2 * i1[self.s_flag]) * np.sin(2 * i2[self.s_flag]) + 2 * cos_Th[self.s_flag] * np.cos(2 * i1[self.s_flag]) * np.cos(2 * i2[self.s_flag])) * self.S[2, self.s_flag])

  def simulate(self, max_steps=None, info=None):
    '''
    Run a simulation.

    Parameters
    max_steps (int): Maximum number of steps before ending simulation
    info (int): Interval in steps at which to print number of steps
    '''
    # run simulation as long as some packets are still evolving
    while np.any(np.logical_not(self.d_flag)):
      self.step_cell()
      self.interact()
      self.write_photons(TERMINATE, self.end_condition * np.logical_not(self.d_flag))
      self.d_flag[self.end_condition] = True
      self.nsteps += 1

      # stop simulation if max steps reached
      if max_steps and self.nsteps >= max_steps:
        self.write_photons(MAX, np.logical_not(self.d_flag))
        percent_done = np.sum(self.d_flag) / self.N * 100
        self.log("Reached max steps with %.3g%% of photons done" % percent_done)
        break

      # print number of steps if required by info
      if info and self.nsteps % info == 0:
        # os.system('clear') # clear output
        percent_done = np.sum(self.d_flag) / self.N * 100
        self.log("%d steps (%.3g%% done)" % (self.nsteps, percent_done))

    self.log("All photons done")

  @property
  def end_condition(self):
    ''' Condition for stopping evolution '''
    outside_grid = (self.pos[0] < self.grid.x_min) + (self.pos[0] > self.grid.x_max) + (self.pos[1] < self.grid.y_min) + (self.pos[1] > self.grid.y_max) + (self.pos[2] < self.grid.z_min) + (self.pos[2] > self.grid.z_max)
    return outside_grid

  @property
  def random(self):
    ''' Generate a random array of shape (N) '''
    return self.rng.random(self.N)

  @property
  def sin_th(self):
    ''' Compute sine of theta '''
    return np.sqrt(1 - self.cos_th**2)

  @property
  def pointing(self):
    ''' Compute pointing as a Cartesian unit vector '''
    return np.array([self.sin_th * np.cos(self.phi), self.sin_th * np.sin(self.phi), self.cos_th])

if __name__ == '__main__':

    n_run = int(sys.argv[1]) # number of run

    # create log file
    log_path = os.path.join('logs', 'run%d.log' % n_run)
    outfile_path = os.path.join('data', 'run%d.npy' % n_run)

    grid = Grid(-8, 8, -8, 8, -8, 8, 200) # create grid
    grid.rho[:] = 300 * np.exp(-grid.r) # define density distribution
    sim = Sim(grid, N=100, do_rayleigh=True, outfile_path=outfile_path, log_path=log_path, albedo=0.9)
    sim.simulate(max_steps=10, info=1)
