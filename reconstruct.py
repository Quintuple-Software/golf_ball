import cv2
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
from scipy.optimize import curve_fit


def func3(t, a, b, c, d):
    #return (((a*t + b)*t + c)*t + d)*t + e
    return ((a*t + b)*t + c)*t + d


def func2(t, a, b, c):
    return (a*t + b)*t + c


def funcx(t, a, b, c, d, e):
    return func3(t,a,b,c,d)

def funcy(t, a, b, c, d, e):
    return func2(t,a,b,c)

def funcz(t, a, b, c, d, e):
    return func2(t,a,b,c)


class Reconstructor:
    
    def __init__(self, path):
        """
        import tracks and bbox
        """
        self.tracks = []
        self.diameter = 11 #cm
        self.focal = 28 #mm
        self.imgs = []

        with open('output.txt') as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                vals = line.split(',')
                bbox = []
                for v in vals[1:5]:
                    bbox.append(int(float(v)))
                track = vals[5] == 'True'
                bbox[0] = bbox[0] + bbox[2] / 2
                bbox[1] = bbox[1] + bbox[3] / 2
                bbox.append(track)
                bbox.append(int(vals[0]))
                self.tracks.append(bbox)

            count = len(self.tracks)
            for i in reversed(range(count)):
                bbox = self.tracks[i]
                if bbox[4]:
                    break
                del self.tracks[i]

        with open(path + 'cam.txt') as f:
            lines = f.readlines()
            vals = []
            for line in lines:
                vals.append(float(line.split()[0]))
            self.cx = vals[0]
            self.cy = vals[1]
            self.diameter = vals[2]
            self.focal = vals[3]

        i = 0
        #for bbox in self.tracks:
        #    filename = path + "{}".format(i+1).zfill(8) + ".jpg"
        #    img = cv2.imread(filename)
        #    self.imgs.append(img)
        #    i = i + 1

    def __filter_depth__(self):
        arr = []
        window_size = 3
        moving_averages = []
        # Loop through the array to consider
        # every window of size 3
        i = 0
        for bbox in self.tracks:
            val = bbox[2]
            arr.append(val)
            i = i + 1
            if i < window_size:
                moving_averages.append(val)
                continue
            
            # Store elements from i to i+window_size
            # in list to get the current window
            window = arr[i - window_size : i]
        
            # Calculate the average of current window
            window_average = round(sum(window) / window_size, 2)
            
            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
            
            # Shift window to right by one position
            #i += 1

        for bbox, ma in zip(self.tracks, moving_averages):
            bbox[2] = ma

    def __convert_3d__(self, bbox):
        # focal = 28mm
        #fovy = 65.5
        #fovx = 46.4
        fovd = 75.4

        width = self.cx * 2
        height = self.cy * 2

        diag_ratio = math.tan(math.radians(fovd) / 2) * 2
        width_ratio = width / math.sqrt(width*width + height*height) * diag_ratio
        fovx = math.degrees(math.atan(width_ratio / 2) * 2)
        
        # [x]   [fx 0 cx][X]
        # [y] = [0 fy cy][Y]
        # [z]   [0 0  1 ][Z]
        fx = fy = width / (math.tan(math.radians(fovx) / 2) * 2)

        # atan2(px, depth) = fov/2
        # tan(fov/2) = px / depth
        # depth = px / tan(fov/2)
        w_px = math.sqrt((bbox[2]*bbox[2]+bbox[3]*bbox[3])/2)
        w_metric = self.diameter

        width_dist = width / w_px * w_metric
        depth = width_dist / (math.tan(math.radians(fovx) / 2) * 2)
        
        # X = (x - cx) / fx
        # Y = (y - cy) / fy
        # Z = z
        z = depth
        x = (bbox[0] - self.cx) / fx * depth
        y = -(bbox[1] - self.cy) / fy * depth
        z = z
        
        return (x,y,z)

    def build(self, filter_depth=False):
        if filter_depth:
            self.__filter_depth__()
        self.pts = []

        for bbox in self.tracks:
            pt_3d = self.__convert_3d__(bbox)
            self.pts.append(pt_3d)

        first_pt = self.pts[0]
        for i in range(len(self.pts)):
            pt = self.pts[i]
            self.pts[i] = [x1-x2 for x1,x2 in zip(pt, first_pt)]
    
    def fit(self):
        count = len(self.pts)
        tdata = np.linspace(0, count - 1, count)
        xdata = np.array([pt[0] for pt in self.pts])
        ydata = np.array([pt[1] for pt in self.pts])
        zdata = np.array([pt[2] for pt in self.pts])

        tdata = np.append(tdata, count*3)
        xdata = np.append(xdata, 0)
        ydata = np.append(ydata, 0)
        zdata = np.append(zdata, zdata[-1]*2.5)

        popt_x, pcov = curve_fit(funcx, tdata, xdata)
        popt_y, _ = curve_fit(funcy, tdata, ydata)
        popt_z, _ = curve_fit(funcz, tdata, zdata)

        ltdata = np.linspace(0, count*3 - 1, count*3)
        self.xdata = funcx(ltdata, *popt_x)
        self.ydata = funcy(ltdata, *popt_y)
        self.zdata = funcz(ltdata, *popt_z)
        self.tdata = ltdata

        #self.pts.append([0, 0, zdata[-1]])
        self.trajectory = []
        for t, x, y, z in zip(tdata, xdata, ydata, zdata):
            self.trajectory.append((t,x,y,z))

    def play(self):
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        
        for img, trk in zip(self.imgs, self.tracks):
            if trk is not None:
                (x, y, w, h) = trk
                draw_img = img.copy()
                draw_img = cv2.rectangle(draw_img,
                    (int(x-w/2),int(y-h/2)),
                    (int(x+w/2), int(y+h/2)),
                    color, thickness)

            cv2.imshow("img", draw_img)
            cv2.waitKey(0)
            
    def dump(self):
        for pt in self.pts:
            print(pt)

    def plot(self):
        #plt.ioff()
        
        #fig = plt.figure()
        #fig, ax = plt.subplots()  # a figure with a single Axes
        fig, axs = plt.subplots(2, 1)  # a figure with a 2x2 grid of Axes

        xline = [pt[0] for pt in self.pts]
        yline = [pt[1] for pt in self.pts]
        zline = [pt[2] for pt in self.pts]

        ax1 = axs[0]
        out = ax1.plot(zline, yline, color='red')
        #ax1.plot(self.zdata, self.ydata)
        ax1.scatter(self.zdata, self.ydata, c=self.tdata)
        ax1.set_xlabel('z -> far')
        ax1.set_ylabel('y -> up')
        ax1.set_title('A plot in the Z-Y plane')

        #ax.text(75, .025, r'$\mu=115,\ \sigma=15$')

        #plt.title('3D Trajectory')

        ax2 = axs[1]
        out = ax2.plot(xline, yline, color='red')
        #ax2.plot(self.xdata, self.ydata)
        ax2.scatter(self.xdata, self.ydata, c=self.tdata)
        ax2.set_xlabel('x -> right')
        ax2.set_ylabel('y -> up')
        ax2.set_title('A plot in the X-Y plane')

        fig.tight_layout()
        plt.axis('equal')

        plt.savefig('plot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        #plt.show()

    def export(self):
        f = open("trajectory.txt", "w")
        for pt in self.pts:
            f.write(f'{pt[0]:.6f},{pt[1]:.6f},{pt[2]:.6f}\n')
        f.close()

if __name__ == '__main__':
    reconstructor = Reconstructor('')
    reconstructor.build(filter_depth=False)
    reconstructor.fit()
    #reconstructor.play()
    #reconstructor.dump()
    reconstructor.plot()
    reconstructor.export()
