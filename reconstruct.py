import cv2
import math
import matplotlib.pyplot as plt
import numpy as np


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
        z = -z
        
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
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        xline = [pt[0] for pt in self.pts]
        yline = [pt[1] for pt in self.pts]
        zline = [pt[2] for pt in self.pts]

        # Data for a three-dimensional line
        ax.plot3D(xline, yline, zline, 'gray')

        # Data for three-dimensional scattered points
        xdata = xline
        ydata = yline
        zdata = zline
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

        ax.set_xlabel('x -> right')
        ax.set_ylabel('y -> up')
        ax.set_zlabel('z -> back')

        plt.title('3D Trajectory')

        plt.savefig('plot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def export(self):
        f = open("trajectory.txt", "w")
        for pt in self.pts:
            f.write(f'{pt[0]:.6f},{pt[1]:.6f},{pt[2]:.6f}\n')
        f.close()

if __name__ == '__main__':
    reconstructor = Reconstructor('')
    reconstructor.build(filter_depth=True)
    #reconstructor.play()
    #reconstructor.dump()
    #reconstructor.plot()
    reconstructor.export()