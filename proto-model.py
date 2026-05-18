"""
A Novel Method for Virtual Real-Time Cumuliform Fluid Dynamics Simulation Using Deep Recurrent Neural Networks
Mathematics-MDPI paper at: https://doi.org/10.3390/math13172746 

    Authors:
    * Carlos Jimenez de Parga 
    * Sergio Calo Oliveira
    
  User manual:
      1) Press the 's' key to start the simulation
      2) Press the 'p' key to PAUSE and RESUME
      3) When the cloud reaches PXs = 30:
          3.1) Press the 'c' key to reset the simulation
          3.2) Press the '3' key to create a new cloud
          3.3) Return to step 2 
  
"""


# For Vispy
import sys
from vispy import app
from vispy import scene
from vispy.visuals.transforms import STTransform

# For Matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math 

# Jos Stam fluid library (treat as a black box)
from solver3D import vel_step, dens_step, M, N, O

# Others
import time

import torch
import modelo
import os
from modelo import device

# Line added to work around a Torch and Vispy compatibility error
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Path to the model stored on your computer
PATH = 'LSTM_beta_4.pth'

# Select the model to use, which must match the one loaded locally above
model_name = 'LSTM'
if model_name == 'LSTM':
    model = modelo.LSTM(input_size=105, output_size=105, hidden_dim=400, n_layers=4)
    
if model_name == 'RNN':
    model = modelo.RNN(input_size=105, output_size=105, hidden_dim=40, n_layers=3)

# Load trained weights
model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
#model.load_state_dict(torch.load(PATH))
model.eval()

model.to(device)
print('Using device', device)



dt = 0.4 # time delta
diff = 0.0 # diffusion
visc = 0.00001 # viscosity
force = 0.2
cd = 0.1
# Wind force

# 3D fluid grid size
size = (M+2)*(N+2)*(O+2)


# Wind force 3D components u = X, v = Y, w = Z
u = np.zeros(size)  # velocity
u_prev = np.zeros(size)
v = np.zeros(size)  # velocity
v_prev = np.zeros(size)
w = np.zeros(size)  # velocity
w_prev = np.zeros(size)

pause = False
iniciado = False

# Presentation layer

# Matplotlib class
class Matplot:
    # Initialise Matplotlib
    def __init__(self):
        fig = plt.figure()
        self.ax = fig.gca(projection='3d')
        self.ax.set_aspect('auto')

        plt.gca().patch.set_facecolor('silver')
        self.ax.w_xaxis.set_pane_color((0.1, 0.2, 0.8, 1.0))
        self.ax.w_yaxis.set_pane_color((0.1, 0.2, 0.8, 1.0))
        self.ax.w_zaxis.set_pane_color((0.1, 0.2, 0.8, 1.0))
    
    # Draw sphere using Matplotlib
    def draw_sphere(self,x,y,z,radius,colour="white",mode = "solid"):
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)

        a = radius * np.outer(np.sin(u), np.sin(v)) + x
        b = radius * np.outer(np.sin(u), np.cos(v)) + y
        c = radius * np.outer(np.cos(u), np.ones_like(v)) + z
  
        if (mode == "wireframe"):
            self.ax.plot_wireframe(a, b, c, color = colour)
        else:
            self.ax.plot_surface(a, b, c, color = colour)  
    
    def show(self):
        plt.show()

# Vispy OpenGL class
class Vispy(app.Canvas):

    # Initialise Vispy
    canvas = scene.SceneCanvas(keys='interactive', bgcolor=[0.6,0.7,0.8], size = [1800,600], show=True)            
    def __init__(self):    
        self.view1 = Vispy.canvas.central_widget.add_view()
        self.cam1 = scene.cameras.TurntableCamera(parent=self.view1.scene,
                                     name='Turntable')
        self.cam2 = scene.cameras.PanZoomCamera(parent=self.view1.scene,
                                     name='Panzoom')
        self.cam1.elevation = 90.0
        self.cam1.azimuth = 0.0
        self.cam1.distance = 20
        self.cam1.roll = 0

        self.cam2.aspect = 2.0
        
        self.view1.camera = self.cam1 
          
        # Create an XYZAxis visual
        self.axis = scene.visuals.XYZAxis(parent=self.view1)
        s = STTransform(translate=(1100, 500), scale=(50, 50, 50, 1))
        affine = s.as_matrix()
        self.axis.transform = affine
        
        canvas.measure_fps()

    # Update coordinate axis    
    def update_axis(self):
        self.axis.transform.reset()

        self.axis.transform.rotate(self.cam1.roll, (0, 0, 1))
        self.axis.transform.rotate(self.cam1.elevation, (1, 0, 0))
        self.axis.transform.rotate(self.cam1.azimuth, (0, 1, 0))

        self.axis.transform.scale((50, 50, 0.001))
        self.axis.transform.translate((1100, 500))
        self.axis.update()
    
    # Enable coordinate axis
    def enable_axis(self, state):
        self.axis.visible = state  

    # Set current camera range
    def set_range(self,x,y,z):
        self.view1.camera.set_range(x,y,z)       
        
    # Show Vispy
    def show(self): 
        app.run()
                    
    # Set 3D camera
    def set_3D_camera(self, a):
        self.view1.camera  = self.cam1
        self.view1.camera.set_range(x = [15+a,75+a], y = [-10,10], z = [-10,10]) 
    
    # Set pan-zoom camera
    def set_pan_camera(self):
        self.view1.camera = self.cam2      
        self.view1.camera.set_range(x = [-15,15], y = [-10,10], z = [-10,10]) 
   
    # Draw sphere using Vispy
    def draw_sphere(self,x,y,z,radius): 
        # Reduce rows, cols and depth for higher performance
        sphere = scene.visuals.Sphere(radius, rows=5, cols=5, depth=5,
                    method='cube', parent=self.view1.scene,
                    edge_color=[0.6,0.6,0.6], color = [0.9,0.9,0.9])

        sphere.transform = STTransform(translate=[x, y, z])

        return sphere
        
    # Start timer
    def start_timer(self):
        # Maximum speed (interval is in seconds)
        self.timer = app.Timer(interval = 0.0, connect=on_main_timer, start=True)    
   
    
# Business layer

# Base cloud class
class Cloud:
    def __init__(self):
        # Sphere (X,Y,Z,R) coordinates
        self._sph_positions = []    

    # Generate cumulus cloud
    def create(self, spheres, size, center, nuX, sigX, nuY, sigY, nuZ, sigZ):
            posX = np.random.normal(nuX, sigX, spheres)
            posY = np.random.normal(nuY, sigY, spheres)
            posZ = np.random.normal(nuZ, sigZ, spheres)
                       
            posX = np.clip(posX,-sigX, sigX)
            posY = np.clip(posY,0.0,sigY)
            posZ = np.clip(posZ,-sigZ,sigZ)
                  
                       
            for i in range(spheres):
                self._sph_positions.append([center[0] + posX[i], center[1] + posY[i], center[2] + posZ[i], 
                 # size * (1.0 - 0.1*math.sqrt(math.pow(posX[i] / (2.0*sigX), 2.0) + math.pow(posY[i] / (2.0*sigY), 2.0) + math.pow(posZ[i] / (2.0*sigZ), 2.0)))])      
                 size * ((1/(sigX*math.sqrt(2*math.pi)))*math.exp(-math.pow((posX[i]-nuX),2)/(2*sigX*sigX)) + 
                  (1/(sigY*math.sqrt(2*math.pi)))*math.exp(-math.pow((posY[i]-nuY),2)/(2*sigY*sigY))+
                  (1/(sigZ*math.sqrt(2*math.pi)))*math.exp(-math.pow((posZ[i]-nuZ),2)/(2*sigZ*sigZ)))])
                  
                  
            # Draw the cumulus cloud
            for i in self._sph_positions:  
                self._draw_sphere(i[0],i[1],i[2],i[3])

 def get_total_spheres(self):
        return len(self._sph_positions)
        
    def get_position(self, index):
        return self._sph_positions[index][0], self._sph_positions[index][1], self._sph_positions[index][2]
    def get_all_positions(self):

        return self._sph_positions
        
    def set_position(self, index, position):    
        self._sph_positions[index] = position
        
    def get_radius(self, index):
        return self._sph_positions[index][3]
        
    def set_radius(self, index, radius):
        self._sph_positions[index][3] = radius
 
    def clear_spheres(self):
        self._sph_positions.clear()
        
    def set_all_positions(self, index, position):
        self._sph_positions[index] = position
        

# Cumulus class        
class Cumulus(Cloud):
    def __init__(self, vispy):
        # List of sphere vertices
        self.__primitives = []
        self.__vispy = vispy
        super().__init__()         
    
    # Draw sphere using vispy
    def _draw_sphere(self, x,y,z,radius):  
         self.__primitives.append(self.__vispy.draw_sphere(x,y,z,radius))    
    
    # Translate sphere using vispy
    def move_sphere(self,index,x,y,z):
        sphere = self.__primitives[index]
        sphere.transform = STTransform(translate=[x, y, z])
        
    def move_all(self,x,y,z):
        sphere = self.__primitives
        sphere.transform = STTransform(translate=[x, y, z])
        
    # Remove spheres from screen
    def clear_spheres(self):
        for i in self.__primitives:
            i.parent = None
        self.__primitives.clear()
        super().clear_spheres()     
  
    # Plot the cumulus using Matplot
    def plot(self):
        matplot = Matplot()
        for i in self._sph_positions:
            matplot.draw_sphere(i[0],i[1],i[2],i[3])
            
        matplot.show()    
        
# Initialize Vispy
canvas = Vispy.canvas       

sft = 0.0

# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    global pause
    global sft
    if event.text == '1': # Set 3D camera
        sft = sft + cum1.getPosition(0)[0]
        print(sft)
        vispy.set_3D_camera(sft)
        #vispy.enable_axis(True)
    elif event.text == '2':
        vispy.set_pan_camera() # Set panzoom camera
        vispy.enable_axis(False)
    elif event.text == '3': # Generate new cumulus
        cum1.clear_spheres()
        cum1.create(35,5.3,[0,0,0],0.0,4.5,0.0,3.0,0.0,4.5)
    elif event.text == '4': # Plot cloud with MatPlot
        cum1.plot()
    elif event.text == 's': # Start simulation
        vispy.start_timer()
        print("Applying west wind")
    elif event.text == 'c': # Reset simulation
        sim_reset()
        print("Fluid simulator reset") 
    elif event.text == 'p': # Pause simulation
        pause = not pause      
        if (pause):
          print("SIMULACION PAUSADA")
    elif event.text == 'l':
        shift = cum1.get_position(0)[0]
        vispy.set_3D_camera(shift)
        vispy.enable_axis(True)  
    elif event.text == 'x': # Exit 
         sys.exit(0)
    
            

# Implement axis connection with 3D camera
@canvas.events.mouse_move.connect
def on_mouse_move(event):
    if event.button == 1 and event.is_dragging:
        vispy.update_axis()

# Calculate the index for a 3D array
def IX(i,j,k):
    return int(((i)+(M+2)*(j) + (M+2)*(N+2)*(k)))

# Apply force of west wind +X
def set_force_source (a, b, c):
    for i in range(size):
        a[i] = b[i] = c[i] = 0.0
           
    for i in range(0, M + 2):
        for j in range(0, N + 2):
            for k in range(0, O + 2):
                a[IX(i,j,k)] = force
           
# Delete simulator data           
def clear_data():
    for i in range(size):
        u[i] = v[i] = w[i] = u_prev[i] = v_prev[i] = w_prev[i] = 0.0

total_data = []
# Move the cloud
# iteration = 0 

#First data for network initialization
def initial_state():
     print('initialising... ')
     pos = cum1.get_all_positions()
     for j in range(7):
         vel_list = []
         
         for i in range(cum1.get_total_spheres()):
          x, y, z = pos[i][0:3]
          radius = cum1.get_radius(i)
          
          px = x + M/2.0
          py = y + N/2.0
          pz = z + O/2.0
         
	  # Maintains consistent values ​​within the range
          px = np.clip(px,0,M) - np.random.random(px.shape)/10.
          py = np.clip(py,0,N) - np.random.random(py.shape)/10.
          pz = np.clip(pz,0,O) - np.random.random(pz.shape)/10.
          
               
          index = IX(px,py,pz)
                          

          random_y = np.random.uniform(-0.03, 0.03, y.shape)
          random_z = np.random.uniform(-0.03, 0.03, z.shape)
          auxU = x + np.clip(u[index], -0.5, 1) / 10.0  # Slow progress in +X
          auxV = y + np.clip(v[index] - random_y, -0.2, 0.2) 
          auxW = z + np.clip( w[index] - random_z - random_y, -0.2, 0.2)
      
          vel_list.append(np.array((u[index],    v[index]-random_y ,    w[index]-random_z)))

                
          if (px <= 1.0 or px >= M):
            #print(f"PX has reached the edge: {px}")
            px = - px
          
          if (py <= 1.0 or py >= N):
            #print(f"PY has reached the edge: {py}")
            py = - py
          
          if (pz <= 1.0 or pz >= O):
            #print(f"PZ has reached the edge: {pz}") 
            pz = - pz
            
         # As long as a component (px, py, pz) reaches a maximum (edge), the cloud stops evolving
	 # correctly
          
         cum1.set_position(i, [auxU, auxV, auxW, radius])
          
         cum1.move_sphere(i, auxU, auxV, auxW)
         total_data.append(vel_list)
         pos[i] =  [auxU, auxV, auxW, radius]
         inp = torch.tensor(total_data)
         set_force_source(u_prev, v_prev, w_prev )
         vel_step ( M,N,O, u, v, w, u_prev, v_prev,w_prev, visc, dt )
     if inp.shape[1] != 35:
         inp = torch.hstack([inp, torch.zeros([inp.shape[0], 35 - inp.shape[1], 3])]).clone()
     print('Ready ')
     return inp
     
def move_cloud(inp): 
     ''' Neural networks operate at fixed input lengths; therefore, when a network is run,
	it must always receive the same dimension in the input tensor. To achieve this, when fewer than
	35 spheres are selected, we fill those tensors with zeros until we reach the appropriate input, 
	signaling to the network that there are no spheres present. However, significantly reducing this number
	seems to affect the network somewhat,

	(you can test this by entering 20 spheres, for example).

	This issue will be addressed in future versions of the network.
     '''

         
        inp = inp.view(1,-1,105)
        # Network prediction
        with torch.no_grad():
             output = model(inp.float())
             
             inp[:,:-1] = inp[:,1:].clone().to(device)
             inp[:,-1] = output.clone().to(device)
             output = output.cpu()
                         
             print('average speed:', output[0, ::3].mean())
             for i in range(cum1.get_total_spheres()):
              
              x, y, z = cum1.get_position(i)
              
              #tx1 = time.time()
                     
              radius = cum1.get_radius(i)
                     
              px = x + M/2.0
              py = y + N/2.0
              pz = z + O/2.0

              ''' Maintains speed values ​​below certain limits; these parameters can be modified to better suit the
              desired behavior
              '''
              
              #dx =  np.clip(output[0, i*3],      0,  4)
              # dy =  np.clip(output[0, i*3 + 1],  -0.6,  0.6)
              # dz =  np.clip(output[0, i*3 + 2],  -0.6,  0.6)
              
              # dx =  output[0, i*3]
              
              dx =  output[0, ::3].mean() + cd*(output[0, i*3]-output[0, ::3].mean())
              dy =  output[0, i*3 + 1]
              dz =  output[0, i*3 + 2]
                                             
              auxU = x + dx / 0 # Slow advance on +X (again, this value 5 can be modified to change the cloud speed)
              auxV = y + dy
              auxW = z + dz

              # Values ​​within range (bounds)
              #auxU =  np.clip(auxU,      -M,  M)
              auxV =  np.clip(auxV,  0,  5)
              auxW =  np.clip(auxW,  0,  5)
              
	      # As long as a component (px, py, pz) reaches a maximum (edge), the cloud stops evolving correctly
                         
              cum1.set_position(i, [auxU, auxV, auxW, radius])
              
              cum1.move_sphere(i, auxU, auxV, auxW)
              tx2 = time.time()
              print('time of get position: ', tx2-tx1)
                   
# Simulation core    

def sim_main():
if (not pause):   
      t1 = time.time()
      set_force_source(u_prev, v_prev, w_prev )
      t2 = time.time()
      #vel_step ( M,N,O, u, v, w, u_prev, v_prev,w_prev, visc, dt )
      t3 = time.time()
      move_cloud(inp)  
      t4 = time.time()
      total = t4 -t1
      vel_time = t3 -t2
      #print('total execution time: ', total)
      #print('network execution time: ', vel_time)
      
     
# Reset simulation
def sim_reset():
    clear_data()

# Timer function
def on_main_timer(up):
  sim_main()
    
    
################ MAIN ################

if __name__ == '__main__' and sys.flags.interactive == 0:
    vispy = Vispy()
    cum1 = Cumulus(vispy)
  
    cum1.create(35,5,[-20,0,0],0.0,4.0,0.0,3.0,0.0,4.0)

    inp = initial_state()
    inp = inp.to(device)
    print('initial:    ', inp.shape)
    
    vispy.show()
