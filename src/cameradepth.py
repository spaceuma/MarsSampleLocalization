import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

DEBUG = False

type_LOCCAM = 0
type_NAVCAM = 1

class CameraDepth():
    def __init__(self, cam_type, width, height, physical_height):     
        if cam_type == type_LOCCAM:
            FOV  = 66
            hor_scale   = 0.608797312
            ver_scale   = 0.47352621
        elif cam_type == type_NAVCAM:
            FOV = 66
            hor_scale   = 0.617618024
            ver_scale   = 0.451233953
        else:
            FOV = 60
            hor_scale   = 0.5
            ver_scale   = 0.5
        
        pixel_size      = 4.65e-6 # um
        focal_length    = 3.8e-3  # mm
        
        self.width  = width
        self.height = height
        self.fx     = focal_length  / pixel_size 
        self.fy     = focal_length  / pixel_size 
        self.Cu     = self.width  * hor_scale
        self.Cv     = self.height * ver_scale
        self.phys_h = physical_height


    def loadDepth(self, depth_file):
        depth_array  = np.loadtxt(depth_file)
        depth_matrix = np.reshape(depth_array, (self.height,self.width))
        
        return depth_matrix


    def calcCameraCoord(self, depth_frame, px_coord):
        u = px_coord[0]
        v = px_coord[1]

        d = depth_frame[v,u]

        x_p = (u - self.Cu)*d / self.fx
        y_p = (v - self.Cv)*d / self.fy
        
        # Round to mm
        x_rounded = np.round(x_p,3)
        y_rounded = np.round(y_p,3)
        d_rounded = np.round(d,3)
        
        # Return real coords sample in meters
        return (x_rounded,y_rounded,d_rounded)


    def rotateRefSystem(self, coord_3d, camera_angle):
        angle_radians = math.radians(camera_angle)
        ref_rotation  = math.pi/2

        x_p = coord_3d[0]
        y_p = coord_3d[1]
        d   = coord_3d[2]

        # Left-hand reference system (meters)
        x_r =   x_p
        y_r =   y_p * math.cos(ref_rotation + angle_radians) + d * math.sin(ref_rotation + angle_radians)
        z_r = - y_p * math.sin(ref_rotation + angle_radians) + d * math.cos(ref_rotation + angle_radians) + self.phys_h

        # Round to mm
        x_rounded = np.round(x_r,3)
        y_rounded = np.round(y_r,3)
        z_rounded = np.round(z_r,3)
        
        return(x_rounded,y_rounded,z_rounded)
    

    def obtain3DGlobalcoord(self, depth_frame, px_coord, camera_angle):
        x_cam, y_cam, z_cam = self.calcCameraCoord(depth_frame,px_coord)
        x_r, y_r, z_r       = self.rotateRefSystem((x_cam, y_cam, z_cam), camera_angle)
        
        return (x_r, y_r, z_r)


    def calcDEM(self, depth_frame, resolution, camera_angle):
        x_list, y_list, z_list = [], [], []
        resolution_multiplier  = int(1/resolution)

        for i in range(0,self.width):
            for j in range(0,self.height):
                x, y, z = self.obtain3DGlobalcoord(depth_frame, (i,j),camera_angle)
                x_list.append(x)
                y_list.append(y)
                z_list.append(z)

        x_offset   = np.abs(np.nanmin(x_list))
        DEM_width  = int((np.nanmax(x_list) + x_offset) * resolution_multiplier) + 1
        DEM_height = int((np.nanmax(y_list))            * resolution_multiplier) + 1

        DEM    = np.empty((DEM_width,DEM_height))
        DEM[:] = np.nan 
        for i in range(0,self.width):
            for j in range(0,self.height):
                x = x_list[i * self.height + j]
                y = y_list[i * self.height + j]
                z = z_list[i * self.height + j]
                if(not math.isnan(x) and not math.isnan(y) and not math.isnan(z)):
                    x_pos = int((x + x_offset) * resolution_multiplier) 
                    y_pos = int((y             * resolution_multiplier))
                    z_pos = z * resolution_multiplier
                    DEM[x_pos][y_pos] = z_pos

        ## Debug ##
        if(DEBUG):
            fig = plt.figure()
            plt.imshow(DEM[:, :])
            #plt.ylim(260,160)
            #plt.xlim(20,120)
            plt.xlabel("y distance (cm)")
            plt.ylabel("x distance + xoffset (cm)")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Elevation (cm)', rotation=270, labelpad=10)
            #plt.clim(0,-10)
            plt.close(fig)

        return DEM, x_offset*resolution_multiplier
    

    def calc3DDistance(self, v0, v1):
        euclidean_distance = math.sqrt((v0[0]-v1[0])**2 + (v0[1]-v1[1])**2 + (v0[2]-v1[2])**2)
        euclidean_distance_rounded = np.round(euclidean_distance,3) # Round to mm
        return euclidean_distance_rounded
