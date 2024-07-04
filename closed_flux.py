import sys
sys.path.append('/users/taoshi11/analysator/')
import pytools as pt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib import cm


def get_bulk(name):
    filepath = '/scratch/project_2000203/3D/FHA/bulk1/'
    filename = filepath+name
    
    f = pt.vlsvfile.VlsvReader(file_name = filename)
    # filename = 'bulk1.0000600.vlsv'

    return f

# Total closed flux
def get_B_flux_closed(f):


    # Sort cellids and connection variable

    connection = f.read_variable('vg_connection')
    cellids = f.read_variable("CellID")
    indexids = cellids.argsort()

    cellids = cellids[indexids]
    connection = connection[indexids]

    # find the cells that vg_connection is 0
    connection_zero_indices = np.where(connection==0)[0]
    cellids_closed = cellids[connection_zero_indices]
    coords_closed = f.read_variable("vg_coordinates",cellids_closed)

    # Determine the cell size
    cellsize_closed = f.read_variable("vg_dx",cellids_closed) # Unit: m

    # Find the coords that close to equatorial plane
    eq_indices = np.where((coords_closed[:,2] < cellsize_closed[:,2]) & (coords_closed[:,2] > 0))[0]
    coords_used = coords_closed[eq_indices][:]
    cellids_used = cellids_closed[eq_indices]
    cellsize_used = cellsize_closed[eq_indices]

    # Bz in the selected cells, and the corresponding flux
    B = f.read_variable("vg_b_vol",cellids_used)
    Bz = B[:,2]
    cell_size = cellsize_used[:,0] * cellsize_used[:,1]
    B_flux = Bz * cell_size # unit: Wb
    return coords_used,B_flux,Bz

# Find the MLT values for each coordinate
def get_MLT_value(coords):
    xx, yy = coords[:, 0], coords[:, 1]
    MLT_values = np.zeros((coords.shape[0]))

    for i in range(len(xx)):
        x, y = xx[i], yy[i]

        if x == 0:
            if y > 0:
                MLT = 18
            elif y < 0:
                MLT = 6
        else:
            angle = np.arctan2(y, x)
            MLT = 12 + angle * (12/np.pi)

        MLT_values[i] = MLT

    return MLT_values



# # Separte the coordinates into different MLT regions
# def MLT_separation(coords):

#     xx,yy = coords[:,0],coords[:,1]


#     # 1. Separate the coords into four quadrants
#     def separate_quadrant(x_domain,y_domain):
#         return np.where(x_domain & y_domain)[0]

#     # 2. Separate each quadrant into two part
#     def separate_mlt(indices_quadrant):
#         indices_lower = []
#         indices_upper = []
#         for i in indices_quadrant:
#             if np.abs(xx[i]) > np.abs(yy[i]):
#                 indices_lower.append(i)
#             else:
#                 indices_upper.append(i)
#         return indices_lower,indices_upper

#     # Indices for each quadrant
#     indices_q1 = separate_quadrant((xx>0),(yy>0)) # MLT 12-18
#     indices_q2 = separate_quadrant((xx<0),(yy>0)) # MLT 18-0
#     indices_q3 = separate_quadrant((xx<0),(yy<0)) # MLT 0-6
#     indices_q4 = separate_quadrant((xx>0),(yy<0)) # MLT 6-12

#     # Separate each quadrant into MLT regions
#     indices_12_15,indices_15_18 = separate_mlt(indices_q1)
#     indices_21_0,indices_18_21 = separate_mlt(indices_q2)
#     indices_0_3,indices_3_6 = separate_mlt(indices_q3)
#     indices_9_12,indices_6_9 = separate_mlt(indices_q4)

#     return {
#         "q1":{
#             "MLT_12_15": indices_12_15,
#             "MLT_15_18": indices_15_18
#         },
#         "q2":{
#             "MLT_21_0": indices_21_0,
#             "MLT_18_21": indices_18_21
#         },
#         "q3":{
#             "MLT_0_3": indices_0_3,
#             "MLT_3_6": indices_3_6
#         },
#         "q4":{
#             "MLT_9_12": indices_9_12,
#             "MLT_6_9": indices_6_9
#         }
#     }

def main():
    I = np.arange(800,1612,1)
    # I = np.arange(800,1600,10)
    # I = np.arange(800,801,1)
    Bflux_total = np.zeros((len(I),24))
    images = []
    for i in range(len(I)):
        bulkname = 'bulk1.'+str(I[i]).rjust(7,'0')+'.vlsv'
        f = get_bulk(bulkname)
        coords_used, B_flux, Bz = get_B_flux_closed(f)
        MLT_values = get_MLT_value(coords_used)

        fluxes_in_sectors, bin_edges = np.histogram(MLT_values, bins=24, range=(0, 24), density=False, weights=B_flux)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        print(bin_centers)

        # Save the Bflux in each time step
        # Bflux_total[i,0] = I[i]
        Bflux_total[i,:] = fluxes_in_sectors

        # # Plot the histogram
        # plt.bar(bin_centers, fluxes_in_sectors, width=2) 
        # plt.xlabel('MLT (Magnetic Local Time)')
        # plt.ylabel('Flux(Wb)')
        # plt.title(f'Histogram of Fluxes in Sectors in {I[i]}s')
        # plt.xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])

        # filename0 = f"../../output/Bflux/Bflux_histogram_{I[i]}s.png"
        # plt.savefig((filename0))
        # images.append(imageio.imread(filename0))
        # plt.close()


        # # plot the bar chart on polar axis
        # # Create a polar plot
        # fig = plt.figure(figsize=(16, 6))
        # ax1 = fig.add_subplot(121, polar=True)

        # # Calculate the width of the bars
        # bar_width = np.deg2rad(360/8)

        # idx = [4,5,6,7,0,1,2,3]
        # fluxes_aranged = fluxes_in_sectors[idx]
        # # bin_centers_aranged = bin_centers[idx]

        # colormap = cm.viridis
        # normalize = plt.Normalize(1e8, 1.6e8)


        # ax1.bar(np.deg2rad(bin_centers * 360 / 24), fluxes_aranged, width=np.radians(25), bottom=0.0, align='center',color=colormap(normalize(fluxes_aranged)))

        # # Set polar labels
        # ax1.set_title('Distribution of Magnetic Flux by MLT Sectors')
        # ax1.set_xticks(np.deg2rad([180, 225, 270, 315, 0, 45, 90, 135]))
        # ax1.set_xticklabels(['0', '3', '6', '9', '12', '15', '18', '21'])

        # ax1.set_rmax(1.6e8)

        # ax1.set_yticks([1.0e8, 1.2e8, 1.4e8, 1.6e8])
        # ax1.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

        # ax1.set_xlabel('MLT (Magnetic Local Time)')

        # # plt.title(f'Histogram of Fluxes in Sectors, t={I[i]}s')

        # cax = fig.add_axes([0.08, 0.1, 0.015, 0.8])  
        # sm = plt.cm.ScalarMappable(cmap=colormap,norm = normalize)
        # sm.set_array([])  
        # cbar = plt.colorbar(sm, cax=cax)

        # cbar.set_label('Flux (Wb)')  

        # # The bz plot on the equatorial plane
        # ax2 = fig.add_subplot(122)
        # Re = 6371000
        # scatter = ax2.scatter(coords_used[:, 0] / Re, coords_used[:, 1] / Re, c=Bz * 10**9, cmap='viridis', s=10)
        # scatter.set_clim([0,300])

        # colorbar_ax2 = fig.add_axes([0.92, 0.1, 0.015, 0.8])
        # colorbar = plt.colorbar(scatter, cax=colorbar_ax2)
        # colorbar.set_label('Bz(nT)')

        # ax2.set_xlim([-30,15])
        # ax2.set_ylim([-20,20])
        # ax2.set_xlabel('X/Re')
        # ax2.set_ylabel('Y/Re')
        # ax2.set_title('Magnetic Field Strength (Bz) in the Equatorial Plane')

        # fig.suptitle(f't = {I[i]}s')



        # # Save the plot
        # filename0 = f"../../output/Bflux/polar_Bz/Bflux_{I[i]}s.png"
        # plt.savefig((filename0))
        # images.append(imageio.imread(filename0))
        # plt.close()


    # np.savetxt('../../output/Bflux/Bflux_total_new.csv',Bflux_total,delimiter=',')
    np.save('../../output/Bflux/Bflux_total_MLT1h.npy', Bflux_total)
    # gif_filename = "../../output/Bflux/polar_Bz/Bflux_polarbar.gif"
    # imageio.mimsave(gif_filename,images,duration=0.2)








if __name__ == "__main__":
    main()
